"""Frozen miRTarDS predictor. Selected on development data before these resplits.

Feature order is AB, AA, BB, each with mean, maximum, population standard
deviation, and biased skew. The four branch weights sum to one.
Names c/a/b refer to cross/miRNA-self/gene-self similarities.
"""
import numpy as np

def transform(X):
    cm = X[:, 0]
    cx = X[:, 1]
    cs = X[:, 2]
    am = X[:, 4]
    ax = X[:, 5]
    ast = X[:, 6]
    bm = X[:, 8]
    bx = X[:, 9]
    bst = X[:, 10]
    sc = np.tanh(X[:, 3] / 3.0)
    sa = np.tanh(X[:, 7] / 3.0)
    sb = np.tanh(X[:, 11] / 3.0)
    ma = ((am == 0.0) & (ax == 0.0) & (ast == 0.0) & (X[:, 7] == 0.0)).astype(np.float64)
    mb = ((bm == 0.0) & (bx == 0.0) & (bst == 0.0) & (X[:, 11] == 0.0)).astype(np.float64)
    va = 1.0 - ma
    vb = 1.0 - mb
    both = va * vb
    da = cm - am
    db = cm - bm
    ta = cx - ax
    tb = cx - bx
    ca = cx - cm
    aa = ax - am
    ba = bx - bm
    qa = np.tanh(da / (0.1 + cs + ast))
    qb = np.tanh(db / (0.1 + cs + bst))
    ha = da / (1.2 - 0.5 * (cm + am))
    hb = db / (1.2 - 0.5 * (cm + bm))
    xa = ta / (1.2 - 0.5 * (cx + ax))
    xb = tb / (1.2 - 0.5 * (cx + bx))
    base = np.column_stack((
        da,
        db,
        0.5 * (da + db),
        am - bm,
        ta,
        tb,
        ca,
        aa,
        ba,
        ca - 0.5 * (aa + ba),
        cs - 0.5 * (ast + bst),
        ast - bst,
        qa,
        qb,
        np.tanh(ta / (0.1 + ca + aa)),
        np.tanh(tb / (0.1 + ca + ba)),
        np.minimum(da, db),
        np.maximum(da, db),
        np.minimum(ta, tb),
        np.maximum(ta, tb),
        da * db,
        ta * tb,
        da * ta,
        db * tb,
        qa * qb,
        (da - db) * (ast - bst),
        sc,
        sa,
        sb,
        cs * sc - ast * sa,
        cs * sc - bst * sb,
        ma,
        mb,
        ma * mb,
        da * va,
        db * vb,
        ha,
        hb,
        xa,
        xb,
        np.minimum(ha, hb),
        np.maximum(ha, hb),
        ha * hb,
        xa * xb,
        (ha - hb) * (am - bm),
    ))
    pa = 0.01 + cs * cs + ast * ast
    pb = 0.01 + cs * cs + bst * bst
    za = np.tanh(da / np.sqrt(pa)) * va
    zb = np.tanh(db / np.sqrt(pb)) * vb
    oa = np.exp(-0.5 * da * da / pa) * va
    ob = np.exp(-0.5 * db * db / pb) * vb
    ra = 2.0 * cs * ast / pa * va
    rb = 2.0 * cs * bst / pb * vb
    ka = np.tanh((da + 0.25 * (cs * sc - ast * sa)) / np.sqrt(pa)) * va
    kb = np.tanh((db + 0.25 * (cs * sc - bst * sb)) / np.sqrt(pb)) * vb
    geometry = np.column_stack((
        za,
        zb,
        oa,
        ob,
        ra,
        rb,
        ka,
        kb,
        np.minimum(za, zb) * both,
        np.maximum(za, zb) * both,
        za * zb,
        oa * ob,
        ra * rb,
        za * oa,
        zb * ob,
        (za - zb) * both,
        (oa - ob) * both,
        (ra - rb) * both,
        (ka - za) * va,
        (kb - zb) * vb,
        za * ra,
        zb * rb,
        (za - zb) * (ast - bst) * both,
        (ka - kb) * (sa - sb) * both,
    ))
    weak = np.minimum(da, db)
    strong = np.maximum(da, db)
    tailweak = np.minimum(ta, tb)
    margin = np.column_stack((
        np.tanh(weak / 0.05) * both,
        np.tanh((weak - 0.05) / 0.05) * both,
        np.tanh((weak + 0.05) / 0.05) * both,
        np.tanh(strong / 0.1) * both,
        np.tanh(tailweak / 0.1) * both,
        np.tanh((weak - 0.5 * (cs + np.maximum(ast, bst))) / 0.1) * both,
        np.tanh((weak + 0.5 * (cs - np.maximum(ast, bst))) / 0.1) * both,
        np.tanh(da * ta / 0.02) * va,
        np.tanh(db * tb / 0.02) * vb,
        np.tanh((da - db) / 0.1) * both,
        np.tanh((ta - tb) / 0.1) * both,
        np.tanh((cs - 0.5 * (ast + bst)) / 0.1) * both,
    ))
    shape_a = np.tanh((da - 0.5 * (cs * sc - ast * sa)) / (0.1 + cs + ast)) * va
    shape_b = np.tanh((db - 0.5 * (cs * sc - bst * sb)) / (0.1 + cs + bst)) * vb
    shape = np.column_stack((
        shape_a,
        shape_b,
        np.minimum(shape_a, shape_b) * both,
    ))
    return np.hstack((X, base, geometry, margin, shape))

def recipe():
    return {'components': [{'kind': 'mlp',
                     'weight': 0.3,
                     'columns': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                     'params': {'hidden_dims': [32, 16],
                                'dropout': 0.1,
                                'lr': 0.001,
                                'weight_decay': 0.0001,
                                'batch_size': 256,
                                'max_epochs': 80,
                                'patience': 12,
                                'pad_to': 14}},
                    {'kind': 'mlp',
                     'weight': 0.2,
                     'params': {'hidden_dims': [24, 12],
                                'dropout': 0.2,
                                'lr': 0.0007,
                                'weight_decay': 0.01,
                                'batch_size': 256,
                                'max_epochs': 80,
                                'patience': 12,
                                'activation': 'tanh'}},
                    {'kind': 'histgb',
                     'weight': 0.2,
                     'columns': [0,
                                 1,
                                 2,
                                 3,
                                 4,
                                 5,
                                 6,
                                 7,
                                 8,
                                 9,
                                 10,
                                 11,
                                 12,
                                 13,
                                 16,
                                 17,
                                 21,
                                 22,
                                 24,
                                 25,
                                 26,
                                 27,
                                 32,
                                 33,
                                 43,
                                 44,
                                 45,
                                 46,
                                 48,
                                 49,
                                 50,
                                 51,
                                 52,
                                 53,
                                 54,
                                 55,
                                 56,
                                 81,
                                 82,
                                 83,
                                 84,
                                 85,
                                 86],
                     'params': {'learning_rate': 0.035,
                                'max_iter': 180,
                                'max_leaf_nodes': 15,
                                'max_depth': 4,
                                'min_samples_leaf': 100,
                                'l2_regularization': 5.0,
                                'max_bins': 128}},
                    {'kind': 'svm',
                     'weight': 0.3,
                     'columns': [0,
                                 1,
                                 2,
                                 4,
                                 5,
                                 6,
                                 8,
                                 9,
                                 10,
                                 12,
                                 13,
                                 14,
                                 15,
                                 16,
                                 17,
                                 18,
                                 19,
                                 20,
                                 21,
                                 22,
                                 23,
                                 24,
                                 25,
                                 26,
                                 27,
                                 28,
                                 29],
                     'params': {'kernel': 'rbf', 'C': 0.5, 'gamma': 0.02}}]}
