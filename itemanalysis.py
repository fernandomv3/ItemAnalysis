import pandas as pd
import numpy as np
import scipy.stats as stats
from girth.unidimensional.dichotomous import rasch_conditional,ability_mle

def score(rsp,clave):
    return (rsp == clave).astype(int)

def polyserial(x,y):
    z = stats.zscore(x)
    N = len(y)
    y = pd.Categorical(y).codes + 1
    _,tau = np.unique(y,return_counts=True)
    tau = stats.norm.ppf(np.cumsum(tau[:-1])/N)
    xy = stats.pearsonr(x,y)[0]
    pop = np.std(y,ddof=1)*np.sqrt((N-1)/N)
    return (xy * pop)/np.sum(stats.norm.pdf(tau))

def item_analysis(df):
    nperson,nitem = df.shape
    scores = df.sum(axis=1)
    mean = scores.mean()
    sd = scores.std()
    item_var = df.var()
    alpha = (nitem/(nitem-1))*(1-item_var.sum()/scores.var())
    pvalues = df.mean()
    def item_stats(x):
        scored_deleted = df.drop(columns=x.name).sum(axis=1)
        pbis = df[x.name].corr(scored_deleted)
        alphad = ((nitem-1)/(nitem-2))*(1-item_var.drop(x.name).sum()/scored_deleted.var())
        bis = scored_deleted.corr(df[x.name],method=polyserial)
        return pbis,bis,alphad
    items = pd.concat([pvalues,df.apply(item_stats).T],axis=1)
    items.columns = ["mean","pbis","bis","alphaIfDel"]
    return {"nperson":nperson,"nitem":nitem,"mean":mean,"sd":sd,"alpha":alpha,"items":items}

def distractor_analysis(df,key):
    scored = (df == key).astype(int)
    scores = scored.sum(axis=1).rename('score')
    q = pd.qcut(scores,q=4,labels=['lower','50%','75%','upper']).rename('q')
    def item_distractor_analysis(x):
        dummys = pd.get_dummies(x)
        n = dummys.sum().rename('n')
        p = dummys.mean().rename('p')
        pbis = dummys.apply(lambda y: y.corr(scores-y)).rename('pbis')
        pq = dummys.join(q).groupby('q').mean().T
        disc = (pq['upper'] - pq['lower']).rename('disc')
        res = pd.concat([n,p,pbis,disc,pq],axis=1).rename_axis('choice')
        res['correct'] = res.index == key[x.name]
        res = res.unstack()
        return res
    res = df.apply(item_distractor_analysis).T.stack()
    res['n'] = res['n'].astype(int)
    return res.reindex(columns=['correct','n','p','pbis','disc','lower','50%','75%','upper'])

def rasch_estimate_cmle(X,anchored_difficulty=None):
    X = X.T
    if anchored_difficulty is None:
        difficulty = rasch_conditional(X)['Difficulty']
    else:
        difficulty = anchored_difficulty
    ability = ability_mle(X.to_numpy(),difficulty,1,no_estimate=5)
    return difficulty,ability


def fit_stats(X,difficulty,ability,axis=0):
    dif,ab = np.meshgrid(difficulty,ability)
    expected =  np.exp(ab-dif)/(1+np.exp(ab-dif))
    variances = np.multiply(expected , 1-expected)
    kurtosis = np.multiply(variances,expected**3 + (1-expected)**3)

    error = np.sqrt(1/np.sum(variances,axis=axis))

    residuals = X - expected

    fit = (residuals**2)/variances

    outfit = np.mean(fit,axis=axis)
    var_out = np.mean((kurtosis/(variances**2))-1,axis=axis)/kurtosis.shape[axis]
    zstdoutfit = (np.cbrt(outfit)-1)*(3/np.sqrt(var_out)) + (np.sqrt(var_out)/3)

    infit = np.sum(residuals**2,axis=axis)/np.sum(variances,axis=axis)
    var_in = np.sum((kurtosis-(variances**2)),axis=axis)/(np.sum(variances,axis=axis)**2)
    zstdinfit = (np.cbrt(infit)-1)*(3/np.sqrt(var_in)) + (np.sqrt(var_in)/3)

    return pd.DataFrame({'error':error,'infit':infit,'zinfit':zstdinfit,'outfit':outfit,'zoutfit':zstdoutfit})

def rasch(X,anchored_difficulty=None):
    resDif = pd.DataFrame(index=X.columns)
    resAb = pd.DataFrame(index=X.index)

    resDif = pd.concat([resDif,X.sum().rename('score')],axis=1)
    resDif["count"] = X.shape[0]

    resAb = pd.concat([resAb,X.sum(axis=1).rename('score')],axis=1)
    resAb["count"] = X.shape[1]

    difficulty,ability = rasch_estimate_cmle(X,anchored_difficulty)
    
    item_fit = fit_stats(X,difficulty,ability,axis=0)
    person_fit = fit_stats(X,difficulty,ability,axis=1)

    dif = pd.DataFrame({'measure':difficulty},index=X.columns).join(item_fit)
    ab = pd.DataFrame({'measure':ability},index=X.index).join(person_fit)

    return resDif.join(dif),resAb.join(ab)

def true_score(difficulty,logits,inf=-5.5,sup=5.5):
    if not logits:
        difficulty = (difficulty-500)/100
    difficulty = difficulty.round(3)
    ni = difficulty.shape[0]
    hab = np.arange(inf,sup + 0.001,0.001)
    d,b = np.meshgrid(difficulty,hab)
    p = np.sum(np.exp(b-d)/(1+np.exp(b-d)),axis=1)
    idx = np.round(p-0.5,0)
    idx = (np.concatenate([[-1],idx]) != np.concatenate([idx,[ni]]))[:-1]
    df = pd.DataFrame({'Escala':hab[idx],'PuntajeAprox':p[idx]})
    last = pd.DataFrame({'Escala':[sup],'PuntajeAprox':[ni]})
    df = pd.concat([df,last],ignore_index=True)
    df['PuntajeAprox'] = df['PuntajeAprox'].round().astype(int)
    if df.iloc[0,1] != 0: #el primer valor no es 0
        df.iloc[0,1] = 0
    return df