"""Configuration parameter extraction from sweep configs."""

import logging
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


class ConfigExtractor:
    """Extracts parameters from configuration files."""

    def __init__(self):
        """Initialize the config extractor."""

    def extract_from_file(self, config_path: Path) -> dict:
        """
        Extract parameters from a config file.

        Args:
            config_path: Path to config YAML file

        Returns:
            Dictionary of extracted parameters
        """
        try:
            config = OmegaConf.load(config_path)
            return self.extract_from_config(config)
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            return {}

    def extract_from_config(self, config) -> dict:
        """
        Extract all relevant parameters from a loaded config.

        Args:
            config: OmegaConf config object

        Returns:
            Dictionary of extracted parameters
        """
        params = {}

        # Extract model parameters
        params.update(self._extract_model_params(config))

        # Extract training parameters
        params.update(self._extract_training_params(config))

        # Extract experiment parameters
        params.update(self._extract_experiment_params(config))

        return params

    def _extract_model_params(self, config) -> dict:
        """Extract model-related parameters."""
        params = {}

        try:
            if hasattr(config, "model") and hasattr(config.model, "encoder"):
                encoder = config.model.encoder
                if hasattr(encoder, "type"):
                    params["encoder_type"] = str(encoder.type).lower()
                if hasattr(encoder, "params"):
                    enc_params = encoder.params
                    if hasattr(enc_params, "router_mode"):
                        params["router_mode"] = enc_params.router_mode
                    if hasattr(enc_params, "pathway_dim"):
                        params["pathway_dim"] = enc_params.pathway_dim

            if hasattr(config, "model") and hasattr(config.model, "core"):
                core = config.model.core
                if hasattr(core, "type"):
                    # Canonical architecture selector (preferred over legacy booleans)
                    core_type = str(core.type).lower()
                    canonical_type = {
                        "matchedtotalparammlp": "total_param_mlp",
                        "matchedactiveparammlp": "active_param_mlp",
                        "einet_sh": "dendritic_shunting",
                        "einet_ns": "dendritic_additive",
                        "einet_sh_flat": "flat_shunting",
                        "einet_ns_flat": "flat_additive",
                        "normadd": "dendritic_normalized_additive",
                        "normalized_additive": "dendritic_normalized_additive",
                    }.get(core_type, core_type)
                    params["network_type"] = canonical_type

                if str(
                    getattr(core, "type", "")
                ).lower() == "population_network" and hasattr(
                    core, "population_network"
                ):
                    params.update(
                        self._extract_population_network_params(core.population_network)
                    )

                target_active_parameters = getattr(
                    core, "target_active_parameters", None
                )
                if target_active_parameters is not None:
                    params["target_active_parameters"] = int(target_active_parameters)

                # Connectivity parameters
                if hasattr(core, "connectivity"):
                    conn = core.connectivity
                    if hasattr(conn, "ee_synapses_per_branch_per_layer"):
                        ee_values = conn.ee_synapses_per_branch_per_layer
                        if isinstance(ee_values, ListConfig):
                            ee_values = OmegaConf.to_container(ee_values, resolve=True)

                        params["ee_value"] = (
                            ee_values[0] if isinstance(ee_values, list) else ee_values
                        )

                    if hasattr(conn, "ie_synapses_per_branch_per_layer"):
                        ie_values = conn.ie_synapses_per_branch_per_layer
                        if isinstance(ie_values, ListConfig):
                            ie_values = OmegaConf.to_container(ie_values, resolve=True)

                        params["ie_value"] = (
                            ie_values[0] if isinstance(ie_values, list) else ie_values
                        )

                # Architecture parameters
                if hasattr(core, "architecture"):
                    arch = core.architecture
                    if hasattr(arch, "excitatory_layer_sizes"):
                        excitatory_layers = self._to_plain_container(
                            arch.excitatory_layer_sizes
                        )
                        params["excitatory_layers"] = excitatory_layers
                        if (
                            isinstance(excitatory_layers, list)
                            and excitatory_layers
                            and len(set(excitatory_layers)) == 1
                        ):
                            params["excitatory_width"] = excitatory_layers[0]
                    if hasattr(arch, "inhibitory_layer_sizes"):
                        inhibitory_layers = self._to_plain_container(
                            arch.inhibitory_layer_sizes
                        )
                        params["inhibitory_layers"] = inhibitory_layers
                        if (
                            isinstance(inhibitory_layers, list)
                            and inhibitory_layers
                            and len(set(inhibitory_layers)) == 1
                        ):
                            params["inhibitory_width"] = inhibitory_layers[0]
                    if hasattr(arch, "excitatory_branch_factors"):
                        branch_factors = arch.excitatory_branch_factors
                        if (
                            isinstance(branch_factors, list)
                            and len(branch_factors) == 1
                        ):
                            params["branch_factors"] = branch_factors[0]
                        else:
                            params["branch_factors"] = "-".join(
                                str(x) for x in branch_factors
                            )

                # Morphology parameters
                saved_use_shunting = None
                if hasattr(core, "morphology"):
                    morph = core.morphology
                    if hasattr(morph, "use_shunting"):
                        saved_use_shunting = bool(morph.use_shunting)
                        params["use_shunting"] = saved_use_shunting
                        if str(getattr(core, "type", "")).lower() in [
                            "point_mlp",
                            "mlp",
                        ]:
                            params["point_reference_use_shunting"] = saved_use_shunting
                    if hasattr(morph, "use_additive_normalization"):
                        params["use_additive_normalization"] = bool(
                            morph.use_additive_normalization
                        )
                    if hasattr(morph, "additive_mode"):
                        params["additive_mode"] = str(morph.additive_mode)

                # If core.type explicitly specifies the network variant, make sure the
                # extracted `use_shunting` matches the *actual* instantiated behavior.
                if "network_type" in params:
                    nt = str(params["network_type"]).lower()
                    inferred_use_shunting = None
                    if nt in ["dendritic_shunting", "flat_shunting"]:
                        inferred_use_shunting = True
                    elif nt in [
                        "dendritic_additive",
                        "dendritic_normalized_additive",
                        "flat_additive",
                        "flat_normalized_additive",
                        "dendritic_mlp",
                        "flat_mlp",
                        "mlp",
                    ]:
                        inferred_use_shunting = False
                    elif nt == "point_mlp":
                        # For point_mlp, morphology.use_shunting does not change the
                        # point-neuron computation. It records whether the baseline was
                        # parameter-matched to a shunting or additive dendritic reference.
                        inferred_use_shunting = None

                    if inferred_use_shunting is not None:
                        if (
                            saved_use_shunting is not None
                            and saved_use_shunting != inferred_use_shunting
                        ):
                            logger.warning(
                                "Config mismatch: network_type='%s' implies "
                                "use_shunting=%s but saved morphology.use_shunting=%s. "
                                "Using network_type for sweep extraction.",
                                nt,
                                inferred_use_shunting,
                                saved_use_shunting,
                            )
                        params["use_shunting"] = inferred_use_shunting

        except Exception as e:
            logger.warning(f"Error extracting model params: {e}")

        network_type = str(params.get("network_type", "")).lower()
        if "use_shunting" in params and network_type not in {"point_mlp", "mlp"}:
            if bool(params["use_shunting"]):
                params["integration_rule"] = "shunting"
            elif bool(params.get("use_additive_normalization", False)):
                params["integration_rule"] = "normalized_additive"
            else:
                params["integration_rule"] = str(params.get("additive_mode", "raw"))

        return params

    def _to_plain_container(self, value):
        """Convert OmegaConf containers to plain Python containers."""
        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
        return value

    def _format_branch_factors(self, branch_factors) -> str | None:
        branch_factors = self._to_plain_container(branch_factors)
        if branch_factors is None:
            return None
        if isinstance(branch_factors, (list, tuple)):
            return "-".join(str(x) for x in branch_factors)
        return str(branch_factors)

    def _summarize_population_field(self, layers: list, field: str):
        """Return one analysis-safe value for an effective population field.

        Population-network options can be declared in each layer's defaults and
        overridden per population.  Sweep analysis must therefore inspect the
        effective value for every instantiated population.  A mixed value is
        encoded explicitly instead of silently selecting the first layer and
        merging scientifically different conditions.
        """
        values = []
        missing = object()
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            defaults = layer.get("population_defaults") or {}
            if not isinstance(defaults, dict):
                defaults = {}
            populations = layer.get("populations") or []
            if not isinstance(populations, list) or not populations:
                value = defaults.get(field, missing)
                if value is not missing:
                    values.append(value)
                continue
            for population in populations:
                if not isinstance(population, dict):
                    continue
                overrides = population.get("population") or {}
                if not isinstance(overrides, dict):
                    overrides = {}
                value = overrides.get(field, defaults.get(field, missing))
                if value is not missing:
                    values.append(value)

        unique = []
        for value in values:
            if value not in unique:
                unique.append(value)
        if not unique:
            return None
        if len(unique) == 1:
            return unique[0]
        return "mixed:" + "|".join(sorted(str(value) for value in unique))

    def _summarize_structured_connectivity(
        self, layers: list
    ) -> tuple[bool | str, str, str]:
        """Summarize the effective structured-routing policy across populations."""
        values: list[tuple[bool, str, str]] = []
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            defaults = layer.get("population_defaults") or {}
            if not isinstance(defaults, dict):
                defaults = {}
            populations = layer.get("populations") or []
            population_records = populations if isinstance(populations, list) else []
            if not population_records:
                population_records = [{}]
            for population in population_records:
                if not isinstance(population, dict):
                    continue
                overrides = population.get("population") or {}
                if not isinstance(overrides, dict):
                    overrides = {}
                declaration = overrides.get(
                    "structured_connectivity",
                    defaults.get("structured_connectivity", {}),
                )
                declaration = self._to_plain_container(declaration) or {}
                if not isinstance(declaration, dict):
                    declaration = {}
                enabled = bool(declaration.get("enabled", False))
                method = str(declaration.get("method", "none")) if enabled else "none"
                spatial = declaration.get("spatial") or {}
                if not isinstance(spatial, dict):
                    spatial = {}
                region_mode = (
                    str(spatial.get("region_mode", "partition"))
                    if enabled
                    and method in {"spatial_morphology", "morphology_spatial"}
                    else "none"
                )
                value = (enabled, method, region_mode)
                if value not in values:
                    values.append(value)

        if not values:
            return False, "none", "none"
        if len(values) == 1:
            return values[0]
        methods = "|".join(
            sorted(
                f"{int(enabled)}:{method}:{region_mode}"
                for enabled, method, region_mode in values
            )
        )
        return "mixed", f"mixed:{methods}", "mixed"

    def _extract_population_network_params(self, population_network) -> dict:
        """Extract sweep axes from canonical ``population_network`` configs.

        Population-network configs encode the same E/I sweep axes as legacy
        EINet configs, but under per-layer ``population_defaults`` and optional
        per-population overrides. For sweep aggregation we use the readout
        population of the first layer, which is the old single-layer E/I case
        and the intended summary target for layered population-network sweeps.
        """
        params = {}
        pn = self._to_plain_container(population_network) or {}
        if not isinstance(pn, dict):
            return params

        layers = pn.get("layers") or []
        if not layers:
            return params

        params["network_layer_count"] = len(layers)
        for field in (
            "blocklinear_strategy",
            "topk_type",
            "topk_strategy",
            "reactivation_strategy",
            "reactivation_init_policy",
            "reactivate",
            "dbl_init_method",
            "init_method",
        ):
            value = self._summarize_population_field(layers, field)
            if value is not None:
                params[field] = value

        structured_enabled, structured_method, spatial_region_mode = (
            self._summarize_structured_connectivity(layers)
        )
        params["structured_connectivity_enabled"] = structured_enabled
        params["structured_connectivity_method"] = structured_method
        params["spatial_region_mode"] = spatial_region_mode
        if structured_enabled is True:
            params["synaptic_routing"] = (
                f"{structured_method}_{spatial_region_mode}"
                if structured_method in {"spatial_morphology", "morphology_spatial"}
                and spatial_region_mode != "partition"
                else structured_method
            )
        elif structured_enabled is False and params.get("topk_type") == "indexed":
            params["synaptic_routing"] = "global_fixed_index"
        elif structured_enabled == "mixed":
            params["synaptic_routing"] = "mixed"
        else:
            params["synaptic_routing"] = "unstructured"

        first_layer = layers[0]
        if not isinstance(first_layer, dict):
            return params

        layer_defaults = first_layer.get("population_defaults") or {}
        if not isinstance(layer_defaults, dict):
            layer_defaults = {}

        populations = first_layer.get("populations") or []
        readout_name = first_layer.get("readout_population")
        readout_pop = None
        if isinstance(populations, list):
            for pop in populations:
                if isinstance(pop, dict) and pop.get("name") == readout_name:
                    readout_pop = pop
                    break
            if readout_pop is None:
                for pop in populations:
                    if isinstance(pop, dict) and pop.get("polarity") == "excitatory":
                        readout_pop = pop
                        break
            if readout_pop is None and populations:
                first_pop = populations[0]
                readout_pop = first_pop if isinstance(first_pop, dict) else None

        pop_overrides = {}
        if isinstance(readout_pop, dict):
            pop_overrides = readout_pop.get("population") or {}
            if not isinstance(pop_overrides, dict):
                pop_overrides = {}

        pop_params = {**layer_defaults, **pop_overrides}

        if "ff_excitatory_synapses" in pop_params:
            params["ee_value"] = pop_params["ff_excitatory_synapses"]
        if "ff_inhibitory_synapses" in pop_params:
            params["ie_value"] = pop_params["ff_inhibitory_synapses"]

        if "use_shunting" in pop_params:
            params["use_shunting"] = bool(pop_params["use_shunting"])
        if "use_additive_normalization" in pop_params:
            params["use_additive_normalization"] = bool(
                pop_params["use_additive_normalization"]
            )
        if "additive_mode" in pop_params:
            params["additive_mode"] = str(pop_params["additive_mode"])

        if isinstance(readout_pop, dict):
            if "branch_factors" in readout_pop:
                formatted = self._format_branch_factors(readout_pop["branch_factors"])
                if formatted is not None:
                    params["branch_factors"] = formatted
            if "n_neurons" in readout_pop:
                params["readout_n_neurons"] = readout_pop["n_neurons"]

        excitatory_sizes = []
        inhibitory_sizes = []
        if isinstance(populations, list):
            for pop in populations:
                if not isinstance(pop, dict) or "n_neurons" not in pop:
                    continue
                polarity = pop.get("polarity")
                if polarity == "excitatory":
                    excitatory_sizes.append(pop["n_neurons"])
                elif polarity == "inhibitory":
                    inhibitory_sizes.append(pop["n_neurons"])
        if excitatory_sizes:
            params["excitatory_layers"] = excitatory_sizes
        if inhibitory_sizes:
            params["inhibitory_layers"] = inhibitory_sizes
            params["inhibitory_population_count"] = len(inhibitory_sizes)
            params["inhibitory_neuron_count"] = sum(
                int(value) for value in inhibitory_sizes
            )

        # Record the actual source of inhibition into the readout population.
        # ``input_mode`` alone is not sufficient: legacy configurations can
        # use mode 0 without constructing inhibitory cells, while canonical
        # PopulationNetwork configurations declare the route explicitly.
        inhibitory_population_names = {
            str(pop.get("name"))
            for pop in populations
            if isinstance(pop, dict) and pop.get("polarity") == "inhibitory"
        }
        connections = first_layer.get("connections") or []
        direct_inhibitory = False
        population_inhibitory = False
        for connection in connections:
            if not isinstance(connection, dict) or not connection.get("enabled", True):
                continue
            if connection.get("target") != readout_name:
                continue
            pathway = str(connection.get("pathway", ""))
            if pathway != "ff_inhibitory":
                continue
            source = str(connection.get("source", ""))
            if source in {"input", "input_i", "inhibitory_input"}:
                direct_inhibitory = True
            if source in inhibitory_population_names:
                population_inhibitory = True

        if direct_inhibitory and population_inhibitory:
            params["inhibitory_routing"] = "mixed_direct_and_population"
        elif population_inhibitory:
            params["inhibitory_routing"] = "population_mediated"
        elif direct_inhibitory:
            params["inhibitory_routing"] = "direct_input"
        else:
            params["inhibitory_routing"] = "none"

        transfer = pn.get("transfer_params") or {}
        if isinstance(transfer, dict):
            if "input_mode" in transfer:
                params["input_mode"] = transfer["input_mode"]
            if "independent_pathways" in transfer:
                params["independent_pathways"] = bool(transfer["independent_pathways"])

        if "recurrent" in first_layer:
            params["population_layer_recurrent"] = bool(first_layer["recurrent"])

        return params

    def _extract_training_params(self, config) -> dict:
        """Extract training-related parameters."""
        params = {}

        try:
            if hasattr(config, "training") and hasattr(config.training, "main"):
                main = config.training.main

                # Strategy
                if hasattr(main, "strategy"):
                    params["strategy"] = main.strategy

                # Common training params
                if hasattr(main, "common"):
                    common = main.common
                    if hasattr(common, "lr") or hasattr(common.param_groups, "lr"):
                        lr = getattr(common, "lr", None) or getattr(
                            common.param_groups, "lr", None
                        )
                        if lr:
                            params["learning_rate"] = lr
                    if hasattr(common, "batch_size"):
                        params["batch_size"] = common.batch_size
                    for source_key, output_key in (
                        ("epochs", "epochs"),
                        ("early_stopping", "early_stopping"),
                        ("patience", "early_stopping_patience"),
                        ("grad_clip_value", "grad_clip_value"),
                        ("weight_decay_rate", "sparse_weight_decay_rate"),
                        ("weight_boosting", "sparse_weight_boosting"),
                    ):
                        if hasattr(common, source_key):
                            params[output_key] = getattr(common, source_key)

                if hasattr(main, "optimizer"):
                    optimizer = main.optimizer
                    if hasattr(optimizer, "name"):
                        params["optimizer_name"] = optimizer.name
                        # OptimizerConfig's runtime default is zero.  Recording
                        # that default distinguishes Adam decay from the custom
                        # sparse maintenance rule in ``common``.
                        params["optimizer_weight_decay"] = getattr(
                            optimizer, "weight_decay", 0.0
                        )

                # Local learning parameters
                if (
                    hasattr(main, "learning_strategy_config")
                    and main.learning_strategy_config
                ):
                    local_cfg = main.learning_strategy_config
                    params.update(self._extract_local_learning_params(local_cfg))

        except Exception as e:
            logger.warning(f"Error extracting training params: {e}")

        return params

    def _extract_local_learning_params(self, local_cfg) -> dict:
        """Extract local learning rule parameters."""
        params = {}

        try:
            if hasattr(local_cfg, "rule_variant"):
                params["rule_variant"] = local_cfg.rule_variant
            if hasattr(local_cfg, "error_broadcast_mode"):
                params["error_broadcast_mode"] = local_cfg.error_broadcast_mode
            if hasattr(local_cfg, "broadcast_rank"):
                params["broadcast_rank"] = local_cfg.broadcast_rank
            if hasattr(local_cfg, "pathway_broadcast_residual"):
                params["pathway_broadcast_residual"] = (
                    local_cfg.pathway_broadcast_residual
                )
            if hasattr(local_cfg, "pathway_activity_gate_strength"):
                params["pathway_activity_gate_strength"] = (
                    local_cfg.pathway_activity_gate_strength
                )
            if hasattr(local_cfg, "encoder_update_mode"):
                params["encoder_update_mode"] = local_cfg.encoder_update_mode
            if hasattr(local_cfg, "decoder_update_mode"):
                params["decoder_update_mode"] = local_cfg.decoder_update_mode
            if hasattr(local_cfg, "update_inactive_weights"):
                params["update_inactive_weights"] = local_cfg.update_inactive_weights

            # Extract from nested four_factor config
            if hasattr(local_cfg, "four_factor"):
                four_f = local_cfg.four_factor
                if hasattr(four_f, "rho_mode"):
                    params["rho_mode"] = four_f.rho_mode
                if hasattr(four_f, "rho_estimator"):
                    params["rho_estimator"] = four_f.rho_estimator

            # Extract from nested five_factor config
            if hasattr(local_cfg, "five_factor"):
                five_f = local_cfg.five_factor
                if hasattr(five_f, "phi_mode"):
                    params["phi_mode"] = five_f.phi_mode
                if hasattr(five_f, "phi_estimator"):
                    params["phi_estimator"] = five_f.phi_estimator

            # Extract from nested morphology_aware config
            if hasattr(local_cfg, "morphology_aware"):
                morph = local_cfg.morphology_aware
                if hasattr(morph, "use_dendritic_normalization"):
                    params["use_dendritic_normalization"] = (
                        morph.use_dendritic_normalization
                    )
                if hasattr(morph, "use_path_propagation"):
                    params["use_path_propagation"] = morph.use_path_propagation
                if hasattr(morph, "path_factor_mode"):
                    params["path_factor_mode"] = morph.path_factor_mode
                if hasattr(morph, "morphology_modulator_mode"):
                    params["morphology_modulator_mode"] = (
                        morph.morphology_modulator_mode
                    )
                if hasattr(morph, "use_branch_role_rules"):
                    params["use_branch_role_rules"] = morph.use_branch_role_rules
                if hasattr(morph, "specialized_branch_scale"):
                    params["specialized_branch_scale"] = morph.specialized_branch_scale
                if hasattr(morph, "mixed_branch_scale"):
                    params["mixed_branch_scale"] = morph.mixed_branch_scale
                if hasattr(morph, "branch_role_power"):
                    params["branch_role_power"] = morph.branch_role_power
                if hasattr(morph, "branch_role_alignment_weight"):
                    params["branch_role_alignment_weight"] = (
                        morph.branch_role_alignment_weight
                    )

            if hasattr(local_cfg, "inhibitory_homeostasis"):
                homeo = local_cfg.inhibitory_homeostasis
                if hasattr(homeo, "enabled"):
                    params["inhibitory_homeostasis_enabled"] = homeo.enabled
                if hasattr(homeo, "mode"):
                    params["inhibitory_homeostasis_mode"] = homeo.mode
                if hasattr(homeo, "weight"):
                    params["inhibitory_homeostasis_weight"] = homeo.weight

            # Extract from nested HSIC config
            if hasattr(local_cfg, "hsic"):
                hsic = local_cfg.hsic
                if hasattr(hsic, "enabled"):
                    params["hsic_enabled"] = hsic.enabled
                if hasattr(hsic, "weight"):
                    params["hsic_weight"] = hsic.weight

        except Exception as e:
            logger.warning(f"Error extracting local learning params: {e}")

        return params

    def _extract_experiment_params(self, config) -> dict:
        """Extract experiment-related parameters."""
        params = {}

        try:
            if hasattr(config, "_sweep_variant"):
                params["sweep_variant"] = str(config._sweep_variant)

            if hasattr(config, "experiment"):
                exp = config.experiment
                if hasattr(exp, "seed"):
                    params["seed"] = exp.seed

            if hasattr(config, "data"):
                data = config.data
                if hasattr(data, "dataset_name"):
                    dataset_name = data.dataset_name
                    params["dataset"] = dataset_name

                    if hasattr(data, "dataset_params"):
                        dataset_params = data.dataset_params
                        if dataset_name in dataset_params:
                            for key, value in dataset_params[dataset_name].items():
                                params[f"{key}"] = value

        except Exception as e:
            logger.warning(f"Error extracting experiment params: {e}")

        return params
