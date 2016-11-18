#!/usr/bin/env python3
import math

# -----------------------------------------------------------------
# **********************************
# Wrap function, because in this place models spectra functions
# are unknown (defined below)
def model_function(modelname):
    known_models = {
        "GM75": J_GM75,
        "BH00": J_BH00,
        "WH03": J_WH03,
        "LA03": J_LA03,
        "WH09": J_WH09,
        # Not yet implemented
        "PAMELA": None,
        # required to rename and to pass right arguments
        # e.g. lambda x: F(x, bla, bla, bla)
        # or return from function
        "F": lambda x: F(x, E0, Phi, Gamma),
    }

    return known_models[modelname]
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# /******************************
# *   1. GM75 model
# */
# ****************************
# This is differential intensity [flux]
# of dim. part. / (cm^2 sr s GeV)
def J_GM75(T_GeV):

    # ****************************
    # J_GM75.a = 9.9E+08 #; for [J] = m^-2 sr s MeV/nucl.
    J_GM75.a = 9.9E+07 #; for [J] = cm^-2 sr s GeV/nucl.
    J_GM75.b = 780.0
    J_GM75.c = -2.5E-04
    # ****************************

    # Convert to MeV/c^2
    TT_MeV = 1000 * T_GeV

    return J_GM75.a * math.pow(
        TT_MeV + J_GM75.b * math.exp(J_GM75.c * TT_MeV),
        -2.65
    )
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# /******************************
# 2. BH00 model [US05?!]
# */
# ****************************
# This is differential intensity [flux]
# of dim. part. / (cm^2 sr s GeV)
def J_BH00(T_GeV):

    # ****************************
    # J_BH00.a = 415.7 #; for [J] = m^-2 sr s MeV/nucl.
    J_BH00.a = 41.57 #; for [J] = cm^-2 sr s GeV/nucl.
    J_BH00.b = 1.0E-07
    J_BH00.c = 1.6488
    J_BH00.M_p = 938.272046 # MeV/c^2
    # ****************************

    # Convert to MeV/c^2
    TT_MeV = 1000 * T_GeV

    return J_BH00.a / (
        J_BH00.b * math.pow(
            TT_MeV * (TT_MeV + 2.0 * J_BH00.M_p),
            1.39) +
        J_BH00.c * math.pow(
            TT_MeV * (TT_MeV + 2.0 * J_BH00.M_p),
            0.135)
    )
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# /******************************
# // 3. WH03 model
# */
# ****************************
# This is differential intensity [flux]
# of dim. part. / (cm^2 sr s GeV)
def J_WH03(T_GeV):

    # ****************************
    # J_WH03.a = 530.0 #; for [J] = m^-2 sr s MeV/nucl.
    J_WH03.a = 53.0 #; for [J] = cm^-2 sr s GeV/nucl.
    J_WH03.b = 1.0E-07
    J_WH03.c = 2.674E-03
    J_WH03.d = 4.919
    # ****************************

    # Convert to MeV/c^2
    TT_MeV = 1000 * T_GeV

    return J_WH03.a / (
        J_WH03.b * math.pow(TT_MeV, 2.8) +
        J_WH03.c * math.pow(TT_MeV, 1.58) +
        J_WH03.d * math.pow(TT_MeV, 0.26)
    )
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# /******************************
# //   4. LA03 model
# */
# ****************************
# This is differential intensity [flux]
# of dim. part. / (cm^2 sr s GeV)
def J_LA03(T_GeV):

    J_LA03.a = 0.823
    J_LA03.b = 0.08
    J_LA03.c = 1.105
    J_LA03.d = 9.202E-02
    J_LA03.e = 22.976
    J_LA03.f = 2.86
    J_LA03.g = 1.5E+03

    T_MeV = 1000 * T_GeV
    lnT = math.log(T_MeV)

    if (T_MeV < 1000.):

        return 0.1 * math.exp( # 0.1 for [J] = cm^-2 sr s GeV/nucl.
            J_LA03.a -
            J_LA03.b * lnT * lnT +
            J_LA03.c * lnT -
            J_LA03.d * math.sqrt(T_MeV)
        )

    else:

        return 0.1 * math.exp( # 0.1 for [J] = cm^-2 sr s GeV/nucl.
            J_LA03.e -
            J_LA03.f * lnT -
            J_LA03.g / T_MeV
        )
# -----------------------------------------------------------------

# -----------------------------------------------------------------
# /******************************
# //   5. WH09 model
# */
# ****************************
# This is differential intensity [flux]
# of dim. part. / (cm^2 sr s GeV)
def J_WH09(T_GeV):

    # ****************************
    # T < 1000 MeV:
    J_WH09.a1 = -124.47673
    J_WH09.b1 = -51.83897
    J_WH09.c1 = 131.64886
    J_WH09.d1 = -241.72524
    J_WH09.e1 = 376.65906
    # T >= 1000 MeV:
    # J_WH09.a2 = 0
    J_WH09.b2 = -51.68612
    J_WH09.c2 = 103.58884
    J_WH09.d2 = -709.70735
    J_WH09.e2 = 1161.55701
    # ****************************

    # Convert to MeV/c^2
    TT_MeV = 1000 * T_GeV
    lnT = math.log(TT_MeV)
    lnlnT = math.log(lnT)

    if (TT_MeV < 1000.):

        return 0.1 * math.exp( # 0.1 for [J] = cm^-2 sr s GeV/nucl.
            J_WH09.a1 +
            J_WH09.b1 * lnlnT * lnlnT +
            J_WH09.c1 * math.sqrt(lnT) +
            J_WH09.d1 / lnT +
            J_WH09.e1 / (lnT * lnT)
        );

    else:

        return 0.1 * math.exp( # 0.1 for [J] = cm^-2 sr s GeV/nucl.
            # J_WH09.a2 +
            J_WH09.b2 * lnlnT * lnlnT +
            J_WH09.c2 * math.sqrt(lnT) +
            J_WH09.d2 / lnT +
            J_WH09.e2 / (lnT * lnT)
        )
# -----------------------------------------------------------------

# -----------------------------------------------------------------
def F(_Ek, _E0, _Phi, _gamma):

    return _Phi * math.pow(_Ek / _E0, _gamma)
# -----------------------------------------------------------------

if __name__ == '__main__':

    for i in range(10):
        print(3. + 0.1*i)

    for i in range(10):
        # print(J_GM75(3. + 0.1*i))
        # print(J_BH00(3. + 0.1*i))
        # print(J_WH03(3. + 0.1*i))
        print(J_LA03(3. + 0.1*i))
        # print(J_WH09(3. + 0.1*i))
