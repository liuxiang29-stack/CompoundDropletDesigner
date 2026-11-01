"""Area-constrained frequency search for the frozen 2D droplet surrogate."""
from functools import lru_cache
import hashlib
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

INLET_WIDTH_M = 50e-6

def frequency_metrics(ai, am, vi, vm, tolerance_percent=10.0):
    """Ai/Am use 1e-9 m2; Qi/Qm are planar fluxes per unit depth (m2/s)."""
    ai, am, vi, vm = np.broadcast_arrays(ai, am, vi, vm)
    qi, qm = vi * INLET_WIDTH_M, 2 * vm * INLET_WIDTH_M
    fi, fm = qi / (ai * 1e-9), qm / (am * 1e-9)
    combined = (qi + qm) / ((ai + am) * 1e-9)
    gap = 200 * np.abs(fi - fm) / (fi + fm)
    return fi, fm, combined, gap, gap <= tolerance_percent + 1e-8

def finite(payload, key, default=None, lo=None, hi=None):
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f'{key} must be a finite number.') from None
    if not math.isfinite(value) or (lo is not None and value < lo) or (hi is not None and value > hi):
        limits = f' [{lo}, {hi}]' if lo is not None or hi is not None else ''
        raise ValueError(f'{key} must be a finite number in the allowed range{limits}.')
    return value

def search(core, payload):
    mode = payload.get('objective', 'area')
    if mode not in ('area', 'max_frequency', 'target_frequency'):
        raise ValueError('Unknown design objective.')
    ai_target = finite(payload, 'ai', lo=1e-8)
    am_target = finite(payload, 'am', lo=1e-8)
    eta0 = finite(payload, 'eta0', lo=1e-9)
    m = finite(payload, 'm', lo=1e-9)
    n = finite(payload, 'n', lo=1e-6, hi=.999999)
    for key, value in [('eta0', eta0), ('m', m), ('n_actual', n)]:
        a, b = core.raw_ranges[key]
        if value < a-1e-10 or value > b+1e-10:
            raise ValueError(f'Inverse design requires {key} within the observed range [{a:g}, {b:g}].')
    requested_count = finite(payload, 'count', 6, lo=1, hi=12)
    if not requested_count.is_integer():
        raise ValueError('Candidate count must be an integer.')
    count = int(requested_count)
    tol_ai = finite(payload, 'ai_tolerance_percent', 5, lo=.05, hi=50)
    tol_am = finite(payload, 'am_tolerance_percent', 5, lo=.05, hi=50)
    freq_consistency = finite(payload, 'consistency_tolerance_percent', 10, lo=.1, hi=100)
    freq_tolerance = finite(payload, 'frequency_tolerance_percent', 5, lo=.05, hi=50)
    max_distance = finite(payload, 'max_support_distance', 2, lo=.05, hi=5)
    min_nonstable = finite(payload, 'min_nonstable_distance', 0, lo=0, hi=3)
    target_frequency = finite(payload, 'target_frequency_hz', 400, lo=.01) if mode == 'target_frequency' else None
    bounds = core.velocity_bounds
    if 'velocity_bounds' in payload:
        custom = payload['velocity_bounds']
        if not isinstance(custom, dict):
            raise ValueError('Velocity bounds must contain vi, vm and vo.')
        bounds = []
        for key in ('vi', 'vm', 'vo'):
            pair = custom.get(key)
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f'Two bounds are required for {key}.')
            a, b = map(float, pair)
            observed = core.raw_ranges[key]
            if not np.isfinite([a,b]).all() or a >= b or a < observed[0]-1e-12 or b > observed[1]+1e-12:
                raise ValueError(f'{key} bounds must be ordered and inside [{observed[0]:g}, {observed[1]:g}] m/s.')
            bounds.append((a,b))
    lower, upper = np.array(bounds).T
    span = upper-lower
    feature_span = np.maximum(core.feature_max-core.feature_min, 1e-8)
    needs_frequency = mode != 'area'

    def evaluate(u):
        v = lower + np.atleast_2d(u)*span
        features = core._features_for_velocities(v, eta0, m, n)
        # Reproduce the frozen soft correction exactly, while rejecting invalid
        # individual rows rather than aborting a whole sampled batch.
        model_features = core.model_features(features)
        raw = np.column_stack([core.corrected_model.inner.predict(model_features),
                               core.corrected_model.shell.predict(model_features)])
        ratio = 2*v[:,1]/v[:,0]
        variances = core.corrected_model.variances
        shift = core.corrected_model.tau*(raw[:,1]-ratio*raw[:,0])/(ratio**2*variances[0]+variances[1])
        areas = raw - np.column_stack([-ratio*variances[0]*shift, variances[1]*shift])
        valid = np.isfinite(areas).all(axis=1) & (areas>0).all(axis=1)
        safe = np.maximum(np.nan_to_num(areas,nan=1e-12,posinf=1e6,neginf=1e-12),1e-12)
        fi,fm,f,gap,consistent = frequency_metrics(safe[:,0],safe[:,1],v[:,0],v[:,1],freq_consistency)
        distance,score = core.support(features)
        nonstable = core.nonstable_distance(features)
        err = 100*(safe/np.array([ai_target,am_target])-1)
        fit = np.sum((err/100)**2,axis=1)
        inside = np.all((features>=core.feature_min-1e-12)&(features<=core.feature_max+1e-12),axis=1)
        qualified = valid & inside & (distance<=max_distance+1e-7) & (nonstable>=min_nonstable-1e-7)
        qualified &= (np.abs(err[:,0])<=tol_ai+1e-6)&(np.abs(err[:,1])<=tol_am+1e-6)
        if needs_frequency:
            qualified &= consistent
        frequency_error = 100*(f/target_frequency-1) if target_frequency is not None else np.zeros(len(f))
        if mode == 'target_frequency':
            qualified &= np.abs(frequency_error)<=freq_tolerance+1e-6
        violations = np.column_stack([
            (tol_ai-np.abs(err[:,0]))/100, (tol_am-np.abs(err[:,1]))/100,
            max_distance-distance, nonstable-min_nonstable,
            *[((features[:,j]-core.feature_min[j])/feature_span[j]) for j in range(5)],
            *[((core.feature_max[j]-features[:,j])/feature_span[j]) for j in range(5)],
            np.where(valid,1.,-100.)])
        if needs_frequency:
            violations = np.column_stack([violations,(freq_consistency-gap)/100])
        return dict(v=v,features=features,areas=areas,fit=fit,err=err,fi=fi,fm=fm,f=f,gap=gap,
                    consistent=consistent,distance=distance,score=score,nonstable=nonstable,
                    qualified=qualified,frequency_error=frequency_error,constraints=violations)

    samples=qmc.Sobol(d=3,scramble=True,seed=20261021).random_base2(14)
    if len(np.unique(samples,axis=0)) != len(samples) or np.any(samples.std(axis=0)<.25):
        raise RuntimeError('Sobol sampling resources are incomplete. Rebuild with the SciPy direction-number data file.')
    sample_sha256=hashlib.sha256(samples.astype('<f8').tobytes()).hexdigest()
    known=np.column_stack([core.data[k] for k in ('vi','vm','vo')])
    samples=np.vstack([samples,np.clip((known-lower)/span,0,1),np.full((1,3),.5),np.zeros((1,3)),np.ones((1,3))])
    initial=evaluate(samples)
    ref_frequency=float(np.median(initial['f']))

    def ordering(e):
        if mode=='max_frequency':
            return np.lexsort((-e['nonstable'],e['distance'],e['fit'],-e['f']))
        if mode=='target_frequency':
            return np.lexsort((-e['nonstable'],e['distance'],e['fit'],np.abs(e['frequency_error'])))
        return np.lexsort((-e['nonstable'],e['distance'],e['fit']))

    # Include both area-fit starts and frequency-oriented feasible starts.
    penalty=initial['fit'] + 20*np.sum(np.minimum(initial['constraints'],0)**2,axis=1)
    indices=list(np.argsort(penalty)[:12])
    feasible=np.flatnonzero(initial['qualified'])
    if len(feasible):
        ranked=ordering({k:v[feasible] for k,v in initial.items()})
        indices=list(feasible[ranked[:12]])+indices
    starts=[]
    for idx in indices:
        if all(np.linalg.norm(samples[idx]-s)>.025 for s in starts):
            starts.append(samples[idx])
        if len(starts)>=10:
            break

    @lru_cache(maxsize=256)
    def cached(t):
        return evaluate(np.array(t))
    def one(u):
        return cached(tuple(map(float,u)))
    def constraints(u):
        return one(u)['constraints'][0]
    def objective(u):
        e=one(u)
        if mode=='max_frequency':
            return -float(e['f'][0])/max(ref_frequency,1e-9)
        if mode=='target_frequency':
            return float(e['frequency_error'][0]/100)**2
        return float(e['fit'][0])
    refined=[]
    for start in starts:
        res=minimize(objective,start,method='SLSQP',bounds=[(0,1)]*3,
                     constraints=[{'type':'ineq','fun':constraints}],
                     options={'maxiter':70,'ftol':1e-10})
        if np.isfinite(res.x).all():
            refined.append(np.clip(res.x,0,1))
    all_u=np.vstack([samples,np.array(refined).reshape(-1,3)])
    final=evaluate(all_u)

    def serialize(i,qualified):
        e=final
        ai,am=map(float,e['areas'][i])
        if not np.isfinite([ai,am]).all() or min(ai,am)<=0:
            return None
        failures=[]
        if abs(e['err'][i,0])>tol_ai+1e-6: failures.append('Inner-area tolerance exceeded')
        if abs(e['err'][i,1])>tol_am+1e-6: failures.append('Shell-area tolerance exceeded')
        if e['distance'][i]>max_distance+1e-7: failures.append('Insufficient data support')
        if not np.all((e['features'][i]>=core.feature_min-1e-12)&(e['features'][i]<=core.feature_max+1e-12)): failures.append('Outside observed descriptor ranges')
        if e['nonstable'][i]<min_nonstable-1e-7: failures.append('Too close to a known nonstable case')
        if needs_frequency and not e['consistent'][i]: failures.append('Phase-frequency consistency failed')
        if mode=='target_frequency' and abs(e['frequency_error'][i])>freq_tolerance+1e-6: failures.append('Target-frequency tolerance exceeded')
        return dict(vi=float(e['v'][i,0]),vm=float(e['v'][i,1]),vo=float(e['v'][i,2]),
                    ai=ai,am=am,ai_error=float(e['err'][i,0]),am_error=float(e['err'][i,1]),
                    combined_error=float(100*np.sqrt(e['fit'][i])),support_score=float(e['score'][i]),
                    support_distance=float(e['distance'][i]),nonstable_distance=float(e['nonstable'][i]),
                    features=e['features'][i].tolist(),frequency_inner_hz=float(e['fi'][i]),
                    frequency_shell_hz=float(e['fm'][i]),frequency_consistency_percent=float(e['gap'][i]),
                    frequency_consistent=bool(e['consistent'][i]),
                    estimated_frequency_hz=float(e['f'][i]) if e['consistent'][i] else None,
                    frequency_error_percent=float(e['frequency_error'][i]) if target_frequency else None,
                    qualified=bool(qualified),failures=failures)

    accepted=[]
    valid_indices=np.flatnonzero(final['qualified'])
    if len(valid_indices):
        ranked=ordering({k:v[valid_indices] for k,v in final.items()})
        for idx in valid_indices[ranked]:
            if any(np.linalg.norm((final['v'][idx]-np.array([r['vi'],r['vm'],r['vo']]))/span)<.012 for r in accepted):
                continue
            row=serialize(idx,True)
            if row: accepted.append(row)
            if len(accepted)>=count: break
    closest=[]
    if not accepted:
        penalties=final['fit']+20*np.sum(np.minimum(final['constraints'],0)**2,axis=1)
        if target_frequency: penalties+=(final['frequency_error']/100)**2
        for idx in np.argsort(penalties):
            row=serialize(idx,False)
            if row and all(np.linalg.norm((final['v'][idx]-np.array([r['vi'],r['vm'],r['vo']]))/span)>.025 for r in closest):
                closest.append(row)
            if len(closest)>=3: break
    return dict(inputs=dict(ai=ai_target,am=am_target,eta0=eta0,m=m,n=n,objective=mode,
                            target_frequency_hz=target_frequency,ai_tolerance_percent=tol_ai,
                            am_tolerance_percent=tol_am,consistency_tolerance_percent=freq_consistency,
                            frequency_tolerance_percent=freq_tolerance,max_support_distance=max_distance,
                            min_nonstable_distance=min_nonstable),
                velocity_bounds={k:[float(a),float(b)] for k,(a,b) in zip(('vi','vm','vo'),bounds)},
                results=accepted,closest_results=closest,
                search=dict(sampled_candidates=len(samples),refined_candidates=len(refined),
                            sobol_sha256=sample_sha256,
                            qualifying_candidates=int(final['qualified'].sum()),
                            strategy='Deterministic Sobol sampling and multistart constrained SLSQP',
                            guarantee='Best identified feasible candidates; not a certificate of a global physical maximum.'),
                frequency_basis='2D phase conservation; equal 50 um inlets; periodic single-core formation; negligible accumulation and bypass.',
                validation_status='Surrogate design candidates. Independent numerical or experimental verification is required.')
