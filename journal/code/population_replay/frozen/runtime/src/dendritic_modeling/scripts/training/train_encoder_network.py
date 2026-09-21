import argparse
import hashlib
import logging
import os
from typing import Optional, Union

import torch

try:
    import torchinfo

    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False
    torchinfo = None


from omegaconf import OmegaConf

from dendritic_modeling.config import (
    DataConfig,
    EncoderConfig,
    EncoderTrainingConfig,
    TrainingConfig,
    load_config,
)
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.networks import BaseNetwork, Identity
from dendritic_modeling.networks.architectures.classical.autoencoder import (
    BaseAutoencoder,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.training import get_trainer
from dendritic_modeling.utils import save_dict, set_seed
from dendritic_modeling.utils.logging_config import LoggerManager

logger = logging.getLogger(__name__)
logger_manager = LoggerManager()


def dict_to_hash_text(
    d: dict, hash_text: str = "", exclude_keys: Optional[list[str]] = None
) -> str:
    if exclude_keys is None:
        exclude_keys = []
    for key, value in d.items():
        if isinstance(value, dict):
            hash_text = dict_to_hash_text(value, hash_text + f"{key}.", exclude_keys)
        else:
            if key not in exclude_keys:
                hash_text += f"{key}{value}"
    return hash_text


def get_load_save_path(
    encoder_network_config: EncoderConfig,
    encoder_train_config: EncoderTrainingConfig,
    task_config: DataConfig,
) -> str:
    encoder_net_config_dict = OmegaConf.to_container(
        encoder_network_config, resolve=True
    )
    encoder_train_config_dict = OmegaConf.to_container(
        encoder_train_config, resolve=True
    )
    task_config_dict = task_config.asdict()
    hash_text = dict_to_hash_text(encoder_net_config_dict)
    hash_text = dict_to_hash_text(
        encoder_train_config_dict,
        hash_text=hash_text,
        exclude_keys=[
            "load_save_root",
            "pretrain_epochs",
            "target_voltage",
            "kl_weight",
            "reverse_training",
            "epochs_per_layer",
            "epochs_per_branch",
            "final_tune_epochs",
        ],
    )
    hash_text = dict_to_hash_text(
        task_config_dict, hash_text=hash_text, exclude_keys=["data_path"]
    )
    hash_id = hashlib.md5(hash_text.encode()).hexdigest()

    load_save_path = os.path.join(encoder_train_config.load_save_root, hash_id)
    logger.info(f"Encoder network hash ID: {hash_id}")
    return load_save_path


def setup_environment(
    encoder_network_config: EncoderConfig,
    train_config: TrainingConfig,
    task_config: DataConfig,
) -> tuple[
    str,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
    Union[BaseNetwork, BaseAutoencoder],
]:
    train_ds, valid_ds, test_ds = get_unified_datasets(
        task_cfg=task_config, train_cfg=train_config
    )
    train_ds: torch.utils.data.Dataset
    valid_ds: torch.utils.data.Dataset
    test_ds: torch.utils.data.Dataset

    logger.info(
        f"Loaded datasets: {len(train_ds)} training, "
        f"{len(valid_ds)} validation, "
        f"{len(test_ds)} test samples"
    )

    input_sample: torch.Tensor = train_ds[0][0]
    encoder_network_config.parameters.input_dim = input_sample.numel()
    encoder_network_config.parameters.input_shape = tuple(input_sample.shape)

    encoder_network = get_architecture(
        encoder_network_config.type,
        OmegaConf.to_container(encoder_network_config.parameters, resolve=True),
    )

    load_save_path = get_load_save_path(
        encoder_network_config=encoder_network_config,
        encoder_train_config=train_config.encoder,
        task_config=task_config,
    )

    logger_manager.set_log_directory(load_save_path)

    log_file = os.path.join(load_save_path, "train.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_file}")

    seed = train_config.seed
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    logger.info(f"Loading dataset: {task_config.dataset}")
    if not hasattr(task_config, "dataset") or not task_config.dataset:
        raise ValueError("No dataset specified in configuration")

    return load_save_path, train_ds, valid_ds, test_ds, encoder_network


def load_train_encoder_network(
    encoder_network: torch.nn.Module,
    encoder_train_config: EncoderTrainingConfig,
    train_ds: torch.utils.data.Dataset,
    valid_ds: torch.utils.data.Dataset,
    test_ds: torch.utils.data.Dataset,
    encoder_network_config: Optional[EncoderConfig] = None,
    task_config: Optional[DataConfig] = None,
    load_save_path: Optional[str] = None,
) -> tuple[
    torch.nn.Module,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
]:
    if isinstance(encoder_network, Identity):
        return encoder_network, train_ds, valid_ds, test_ds

    if not isinstance(encoder_network, BaseAutoencoder):
        # Plain encoders (for example pathway_router) should be trained jointly with
        # the main model rather than pre-encoded into the dataset.
        return encoder_network, train_ds, valid_ds, test_ds
    else:
        train_input, train_label = train_ds[:]
        valid_input, valid_label = valid_ds[:]
        test_input, test_label = test_ds[:]

        train_input: torch.Tensor
        valid_input: torch.Tensor
        test_input: torch.Tensor
        train_label: torch.Tensor
        valid_label: torch.Tensor
        test_label: torch.Tensor

        if load_save_path is None:
            load_save_path = get_load_save_path(
                encoder_network_config=encoder_network_config,
                encoder_train_config=encoder_train_config,
                task_config=task_config,
            )

        model_path = os.path.join(load_save_path, "best_model.pt")
        if os.path.exists(model_path):
            encoder_network.load_state_dict(torch.load(model_path))
            logger.info(f"Loaded model from {model_path}")
        else:
            auto_train_ds = torch.utils.data.TensorDataset(train_input, train_input)
            auto_valid_ds = torch.utils.data.TensorDataset(valid_input, valid_input)

            os.makedirs(load_save_path, exist_ok=True)
            # Use 245MR-compatible structure
            encoder_train_config.trainer.save_path = load_save_path
            encoder_trainer_config_dict = OmegaConf.to_container(
                encoder_train_config.trainer, resolve=True
            )

            optimizer = torch.optim.Adam(
                encoder_network.parameters(),
                lr=getattr(encoder_train_config, "lr", 0.001),
            )

            encoder_trainer = get_trainer(
                strategy="standard",
                optimizer=optimizer,
                trainer_config_dict=encoder_trainer_config_dict,
                analysis_manager=None,
            )
            encoder_trainer.train(
                model=encoder_network,
                train_data=auto_train_ds,
                valid_data=auto_valid_ds,
            )

        encoder_network = encoder_network.cpu()

        with torch.no_grad():
            encoded_train_input = encoder_network.encoder(train_input)
            encoded_valid_input = encoder_network.encoder(valid_input)
            encoded_test_input = encoder_network.encoder(test_input)

        encoded_train_ds = torch.utils.data.TensorDataset(
            encoded_train_input, train_label
        )
        encoded_valid_ds = torch.utils.data.TensorDataset(
            encoded_valid_input, valid_label
        )
        encoded_test_ds = torch.utils.data.TensorDataset(encoded_test_input, test_label)

        for param in encoder_network.parameters():
            param.requires_grad = False

        return (encoder_network, encoded_train_ds, encoded_valid_ds, encoded_test_ds)


def encoder_network_analysis(
    encoder_network: BaseNetwork,
    train_ds: torch.utils.data.Dataset,
    valid_ds: torch.utils.data.Dataset,
    test_ds: torch.utils.data.Dataset,
    load_save_path: str,
):
    """
    Analyze encoder network performance and characteristics.

    Args:
        encoder_network: Trained encoder network
        train_ds: Training dataset
        valid_ds: Validation dataset
        test_ds: Test dataset
        load_save_path: Path to save analysis results
    """
    logger.info("Running encoder network analysis")

    try:
        # Basic encoder analysis - can be expanded based on encoder type
        if hasattr(encoder_network, "encode"):
            # For autoencoder-type networks
            logger.info("Analyzing autoencoder performance")

            # Sample some data for analysis
            sample_data = []
            for i, (data, _) in enumerate(test_ds):
                sample_data.append(data)
                if i >= 10:  # Analyze first 10 samples
                    break

            if sample_data:
                import torch

                sample_batch = torch.stack(sample_data)

                # Get encoded representations
                with torch.no_grad():
                    encoded = encoder_network.encode(sample_batch)
                    if hasattr(encoder_network, "decode"):
                        reconstructed = encoder_network.decode(encoded)

                        # Calculate reconstruction error
                        mse_error = torch.mean(
                            (sample_batch - reconstructed) ** 2
                        ).item()
                        logger.info(f"Mean reconstruction error: {mse_error:.4f}")

                    # Log encoding statistics
                    logger.info(f"Encoded representation shape: {encoded.shape}")
                    logger.info(f"Encoded mean: {torch.mean(encoded).item():.4f}")
                    logger.info(f"Encoded std: {torch.std(encoded).item():.4f}")

        elif hasattr(encoder_network, "forward"):
            # For general transformation networks
            logger.info("Analyzing transformation network")

            # Basic forward pass analysis
            sample_data = []
            for i, (data, _) in enumerate(test_ds):
                sample_data.append(data)
                if i >= 5:
                    break

            if sample_data:
                import torch

                sample_batch = torch.stack(sample_data)

                with torch.no_grad():
                    output = encoder_network(sample_batch)
                    logger.info(f"Output shape: {output.shape}")
                    logger.info(
                        f"Output range: [{torch.min(output).item():.4f}, {torch.max(output).item():.4f}]"
                    )

        else:
            logger.info("Identity encoder - no analysis needed")

        logger.info("Encoder network analysis completed")

    except Exception as e:
        logger.error(f"Error during encoder analysis: {e}")
        logger.info("Encoder analysis failed but continuing...")


def main(config_path: str):
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    encoder_network_config = config.model.encoder
    encoder_train_config = config.training.encoder

    load_save_path, train_ds, valid_ds, test_ds, encoder_network = setup_environment(
        encoder_network_config=encoder_network_config,
        train_config=config.training,
        task_config=config.data,
    )

    logger.info("Network architecture summary before training:")
    try:
        sample_input: torch.Tensor = train_ds[0][0]
        if TORCHINFO_AVAILABLE:
            torchinfo.summary(
                encoder_network,
                input_size=(100, *sample_input.shape),
                col_names=["input_size", "output_size", "num_params"],
                col_width=25,
                row_settings=["depth"],
                depth=7,
            )
            logger.info("Model summary generated (see console output)")
        else:
            logger.warning("torchinfo is not available. Model summary not generated.")
    except Exception as e:
        logger.error(f"Failed to generate model summary before training: {e}")

    encoder_network = load_train_encoder_network(
        encoder_network=encoder_network,
        encoder_train_config=encoder_train_config,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        load_save_path=load_save_path,
    )[0]

    nparams = 0
    for param in encoder_network.parameters():
        nparams += param.numel()
    save_dict({"nparams": nparams}, load_save_path, "nparams")

    encoder_network_config_dict = OmegaConf.to_container(
        encoder_network_config, resolve=True
    )
    encoder_train_config_dict = OmegaConf.to_container(
        encoder_train_config, resolve=True
    )
    task_config_dict = config.task.asdict()
    save_dict(
        {
            "encoder_network": encoder_network_config_dict,
            "encoder_train": encoder_train_config_dict,
            "task": task_config_dict,
        },
        load_save_path,
        "config.json",
    )

    encoder_network_analysis(
        encoder_network=encoder_network,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        load_save_path=load_save_path,
    )

    logger.info("Encoder network training completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an encoder network.")
    parser.add_argument("config", help="Path to the configuration YAML file.")
    args = parser.parse_args()
    main(args.config)
