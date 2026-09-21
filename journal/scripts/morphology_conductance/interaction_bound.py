#!/usr/bin/env python3
"""Independent population interaction bound for the planted conductance family.

The exact formula is an orthogonal functional-ANOVA lower bound for every
function additive across the student's two proximal input blocks. Such additive
functions include the entire permitted conductance model, so the bound applies
for any conductances. Population lognormal moments are evaluated by converged
Gauss-Hermite quadrature, separately from all training and selection.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss

import model
import run as study


SELF=Path(__file__).resolve()


def moments(seed,order):
    nodes,weights=hermgauss(order);z=np.sqrt(2)*nodes;w=weights/np.sqrt(np.pi)
    activity=np.exp(.7*z-.5*.7**2)
    proximal=np.exp(1.2*z-.5*1.2**2)
    g=np.exp(model.teacher_parameters(seed))
    rm=[];rv=[];re=[];ri=[];am=[];av=[];ap=[]
    joint=w[:,None]*w[None,:]
    for leaf in range(4):
        e=g[leaf]*activity[:,None];inh=g[4+leaf]*activity[None,:]
        response=e/(1+e+inh)
        mean=float(np.sum(joint*response));variance=float(np.sum(joint*(response-mean)**2))
        rm.append(mean);rv.append(variance)
        re.append(float(np.sum(joint*response*z[:,None])))
        ri.append(float(np.sum(joint*response*z[None,:])))
    for parent in range(2):
        attenuation=1/(1+g[10+2*parent:12+2*parent].sum()+g[8+parent]*proximal)
        mean=float(w@attenuation)
        am.append(mean);av.append(float(w@(attenuation-mean)**2));ap.append(float(w@(attenuation*z)))
    root=g[14:16]/(1+g[14:16].sum())
    mean=0.;variance=0.
    for p in range(2):
        coupling=g[10+2*p:12+2*p]
        sm=float(coupling@np.asarray(rm[2*p:2*p+2]))
        sv=float((coupling**2)@np.asarray(rv[2*p:2*p+2]))
        mean+=root[p]*am[p]*sm
        variance+=root[p]**2*((av[p]+am[p]**2)*sv+av[p]*sm**2)
    return dict(g=g,terminal_means=np.array(rm),terminal_variances=np.array(rv),
        proximal_attenuation_means=np.array(am),proximal_attenuation_variances=np.array(av),
        terminal_e_hermite=np.array(re),terminal_i_hermite=np.array(ri),proximal_hermite=np.array(ap),
        root=root,population_mean=mean,population_variance=variance)


def lower_bound(seed,task_group,student_group,order):
    m=moments(seed,order)
    permutation=model.GROUPINGS[task_group].ravel()
    student_parent=np.zeros(4,dtype=int)
    for p in range(2):student_parent[model.GROUPINGS[student_group,p]]=p
    anova=0.;hermite=0.;missing=[]
    for canonical_leaf in range(4):
        semantic_leaf=int(permutation[canonical_leaf]);true_parent=canonical_leaf//2
        if student_parent[semantic_leaf]==true_parent:continue
        amplitude=m['root'][true_parent]*m['g'][10+canonical_leaf]
        term=amplitude**2*m['terminal_variances'][canonical_leaf]*m['proximal_attenuation_variances'][true_parent]
        anova+=term
        hermite+=amplitude**2*(m['terminal_e_hermite'][canonical_leaf]**2+m['terminal_i_hermite'][canonical_leaf]**2)*m['proximal_hermite'][true_parent]**2
        missing.append(dict(terminal_input=semantic_leaf,target_parent=true_parent,student_parent=int(student_parent[semantic_leaf]),component_variance=float(term)))
    return dict(seed=seed,task_group=task_group,student_group=student_group,compatible=task_group==student_group,
        quadrature_order=order,population_target_mean=m['population_mean'],population_target_variance=m['population_variance'],
        additive_subtree_mse_lower_bound=anova,additive_subtree_nmse_lower_bound=anova/m['population_variance'],
        first_hermite_mse_lower_bound=hermite,first_hermite_nmse_lower_bound=hermite/m['population_variance'],
        n_omitted_interaction_terms=len(missing),components=missing)


def freeze():
    assert not list((study.OUT/'runs/fresh').glob('*')),'Define diagnostic before fresh outcome inspection'
    study.write(study.OUT/'interaction_bound_protocol.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        status='Independent mechanism diagnostic frozen before fresh outcomes; not a calibration-only selector',
        script_sha256=study.sha(SELF),model_sha256=study.sha(model.__file__),seeds=list(range(15400,15420)),
        quadrature_orders=[32,64,128],primary_reported_order=128,
        exact_formula='Sum over terminals assigned to the wrong proximal block of (root transfer * child conductance)^2 Var(terminal response) Var(proximal inverse total conductance)',
        proof='Orthogonal centered terminal-response × proximal-attenuation components cannot occur in any function additive across student proximal input blocks',
        scope='Lower bound for all allowed conductances, not proof the rational learner attains the bound; numerical moments via quadrature with convergence reported',
        no_training_or_selector_changes=True))


def run():
    protocol=json.loads((study.OUT/'interaction_bound_protocol.json').read_text())
    assert study.sha(SELF)==protocol['script_sha256']
    assert study.sha(model.__file__)==protocol['model_sha256']
    rows=[];components=[]
    for seed in protocol['seeds']:
        for task in range(3):
            for candidate in range(3):
                for order in protocol['quadrature_orders']:
                    result=lower_bound(seed,task,candidate,order)
                    items=result.pop('components');rows.append(result)
                    if order==protocol['primary_reported_order']:
                        components.extend(dict(seed=seed,task_group=task,student_group=candidate,**item) for item in items)
    data=pd.DataFrame(rows);dest=study.OUT/'interaction_bound';dest.mkdir(exist_ok=True)
    data.to_csv(dest/'quadrature_bounds.csv',index=False);pd.DataFrame(components).to_csv(dest/'omitted_components.csv',index=False)
    primary=data[data.quadrature_order.eq(128)].copy();primary.to_csv(dest/'population_bounds.csv',index=False)
    wide=data.pivot(index=['seed','task_group','student_group'],columns='quadrature_order',values='additive_subtree_nmse_lower_bound')
    differences=abs(wide[128]-wide[64])
    study.write(dest/'report.json',dict(n_seeds=len(protocol['seeds']),n_task_candidate_pairs=len(primary),
        min_incompatible_nmse_lower_bound=float(primary[~primary.compatible].additive_subtree_nmse_lower_bound.min()),
        mean_incompatible_nmse_lower_bound=float(primary[~primary.compatible].additive_subtree_nmse_lower_bound.mean()),
        max_order64_vs128_nmse_bound_difference=float(differences.max()),
        max_order32_vs64_nmse_bound_difference=float(abs(wide[64]-wide[32]).max()),
        all_compatible_bounds_zero=bool((primary[primary.compatible].additive_subtree_nmse_lower_bound==0).all()),
        hermite_bound_never_exceeds_anova=bool((primary.first_hermite_mse_lower_bound<=primary.additive_subtree_mse_lower_bound+1e-15).all()),
        protocol_sha256=study.sha(study.OUT/'interaction_bound_protocol.json'),
        scope='Exact analytic bound; numerical expectation evaluated by converged quadrature, not a formally certified integration error'))
    print(json.dumps(json.loads((dest/'report.json').read_text()),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run']);args=parser.parse_args()
    if args.command=='freeze':freeze()
    else:run()
