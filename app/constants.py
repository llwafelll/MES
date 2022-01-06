from txt_importer import dl

# CONSTANTS
CONSTANTS = [
    # SIM_TIME := dl.data["SimulationTime"],
    SIM_TIME := dl.data.get("SimulationTime", None),
    # dT := dl.data["SimulationStepTime"],
    dT := dl.data.get("SimulationStepTime", None),
    K := dl.data["Conductivity"],
    ALPHA := dl.data["Alfa"],
    T_o := dl.data["Tot"],
    t_0 := dl.data["InitialTemp"],
    rho := dl.data["Density"],
    C_p := dl.data["SpecificHeat"],
]


if dl.data == None:
    t_0 = 100 # initial temperature
    SIM_TIME = 500 # simulation time
    dT = 50 # simulation step time
    T_o = 1200 # ambient temperature
    ALPHA = 300 # W/m^2K
    # ALPHA = 25
    C_p = 700 # specific heat
    rho = 7800 # density
    K = 25 # conductivity W/mK
    # K = 30 # first

    # H = .025
    # B = .025
    # N_H = 2
    # N_B = 2

    H = .1
    B = .1
    N_H = 4
    N_B = 4