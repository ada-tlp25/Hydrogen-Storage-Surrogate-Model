from preprocessing.ng_loader import df_all_cycles


# The Initial_Pressure_buildup script is the initial schedule script for the conversion
# from natural gas to hydrogen storage from Benchmark study for the simulation of
# Underground Hydrogen Storage operations Hogeweg et al. (2022)

Initial_Pressure_buildup = r"""
FBHPDEF
50 1000  /

WELSPECS
  WELL3 GROUP1 31 34 1210 GAS 0 STD SHUT YES 1* SEG 3* STD /
  /

WLIST
  '*WELLS F' NEW
  WELL3 /
  /

COMPDAT
  WELL3 31 34 2 2 OPEN 1* 1.89577 0.1905 164.316 0 1* Z 9.90003 /
  WELL3 31 34 3 3 OPEN 1* 1.47081 0.1905 127.482 0 1* Z 9.89976 /
  WELL3 31 34 4 4 OPEN 1* 1.46764 0.1905 127.207 0 1* Z 9.89976 /
  WELL3 31 34 5 5 OPEN 1* 4.4673 0.1905 387.202 0 1* Z 9.89973 /
  WELL3 31 34 6 6 OPEN 1* 4.46731 0.1905 387.203 0 1* Z 9.89972 /
  WELL3 31 34 7 7 OPEN 1* 5.35562 0.1905 464.197 0 1* Z 9.89971 /
  WELL3 31 34 8 8 OPEN 1* 5.34435 0.1905 463.22 0 1* Z 9.89971 /
  WELL3 31 34 9 9 OPEN 1* 10.2979 0.1905 892.565 0 1* Z 9.89971 /
  WELL3 31 34 10 10 OPEN 1* 10.2982 0.1905 892.59 0 1* Z 9.8997 /
  WELL3 31 34 11 11 OPEN 1* 1.56672 0.1905 135.796 0 1* Z 9.89974 /
  WELL3 31 34 12 12 OPEN 1* 1.56673 0.1905 135.796 0 1* Z 9.89974 /
  /

GRUPTREE
  GROUP1 FIELD /
  /

WCONINJE
  WELL3 GAS 1* BHP 2* 90 /
  /
TUNING
0.1 5 /
/
/
WINJGAS
'WELL3' 'STREAM' 'WSTR1' 2* /
 /
TSTEP
11*5 /

TUNING
0.1 5 /
/
/
WELLSHUT
'WELL3' /
 /
TSTEP
6*5 /

WCONINJE
  WELL3 GAS 1* BHP 2* 94 /
  /
TUNING
0.1 5 /
/
/
WELLOPEN
'WELL3' /
 /
WINJGAS
'WELL3' 'STREAM' 'WSTR1' 2* /
 /
TSTEP
12*5 /

TUNING
0.1 5 /
/
/
WELLSHUT
'WELL3' /
 /
TSTEP
5*5 /

WCONINJE
  WELL3 GAS 1* BHP 2* 98 /
  /
TUNING
0.1 5 /
/
/
WELLOPEN
'WELL3' /
 /
WINJGAS
'WELL3' 'STREAM' 'WSTR1' 2* /
 /
TSTEP
12*5 /

TUNING
0.1 5 /
/
/
WELLSHUT
'WELL3' /
 /
TSTEP
5*5 /

WCONINJE
  WELL3 GAS 1* BHP 2* 102 /
  /
TUNING
0.1 5 /
/
/
WELLOPEN
'WELL3' /
 /
WINJGAS
'WELL3' 'STREAM' 'WSTR1' 2* /
 /
TSTEP
12*5 /

TUNING
0.1 5 /
/
/
WELLSHUT
'WELL3' /
 /
TSTEP
6*5 /
"""

Data_script = r"""
WATER

PETOPTS
INITNNC /

MONITOR

UNIFSAVE

UNIFOUT

METRIC

GASWAT

DIFFUSE

AIM

COMPS
  8 /

DIMENS
  61 61 12 /

TABDIMS
  1 1 100 /

GRID

GRIDFILE
 0 1 /

INCLUDE
'Inc_Files\UHS_Benchmark_GRID.INC' /

NOECHO

INCLUDE
'Grid_Props\UHS_Benchmark_GRID.GRDECL' /

INCLUDE
'Grid_Props\UHS_Benchmark_PROP_PERMX.GRDECL' /

INCLUDE
'Grid_Props\UHS_Benchmark_PROP_PERMY.GRDECL' /

INCLUDE
'Grid_Props\UHS_Benchmark_PROP_PERMZ.GRDECL' /

INCLUDE
'Grid_Props\UHS_Benchmark_PROP_PORO.GRDECL' /

ECHO

EDIT

PROPS

INCLUDE
'Inc_Files\UHS_Benchmark_PROPS.INC' /

REGIONS

NOECHO

INCLUDE
'Grid_Props\UHS_Benchmark_PROP_ROCKNUM.GRDECL' /

INCLUDE
'Grid_Props\UHS_Benchmark_PROP_EOSNUM.GRDECL' /

INCLUDE
'Grid_Props\UHS_Benchmark_PROP_PVTNUM.GRDECL' /

INCLUDE
'Grid_Props\UHS_Benchmark_PROP_EQLNUM.GRDECL' /

INCLUDE
'Grid_Props\UHS_Benchmark_PROP_SATNUM.GRDECL' /

ECHO

SOLUTION

EQUIL
 1210 1* 1210 0.1 0 0 1* 1* 0 2 1*
/


RPTRST
 'BASIC=2' 'PRES' 'SGAS' 'XMF' 'YMF' 'ZMF' 'DENG' 'VGAS' /


SUMMARY

INCLUDE
'Inc_Files\UHS_Benchmark_SUM.INC' /

SCHEDULE

"""


def initial_script_named(site):
    """
    Generate Initial Schedule script including initial pressure buildup for a
    given site.

    Parameters:
    ----------
    site (str): The name of the site for which to generate the script.

    Returns:
    -------
    str: The generated Initial Schedule script for the given site.
    """
    Site_title = f""" -- Underground Hydrogen Storage script for {site} \n"""
    return Site_title + Initial_Pressure_buildup


def generate_DATA_script_title(site):
    """
    Generate the initial DATA script for a given site.

    Parameters:
    ----------
    site (str): The name of the site for which to generate the DATA script.

    Returns:
    -------
    str: The generated initial DATA script for the given site.
    """
    Site_title = f""" -- Underground Hydrogen Storage script for {site} \n"""
    return Site_title + Data_script


HPC_schedule_directory = "Schedule_files"


def find_set_schedule(site):
    """
    Adding the file path for the schedule file to the DATA script for a given
    site.

    Parameters:
    ----------
    site (str): The name of the site for which to find the schedule file.

    Returns:
    -------
    str: The generated string to include the schedule file in the DATA script
    for the given site
    """
    script_name = f"{site}_SCH.INC"
    Script_directory = f"INCLUDE\n'{HPC_schedule_directory}/{script_name}' /"
    return Script_directory


# This is the composition used in the benchmark study.
default_comp = """WELLSTRE
'WSTR1' 0 0.17522 0.01818 0.00504 0.00144 0.00008 0.00004 0.8 /
 /

"""
# This is the composition used in this study it is 100% hydrogen.
Full_H2_comp = """WELLSTRE
'WSTR1' 0 0 0 0 0 0 0 1 /
 /

"""


def injection_comp(Comp=None):
    """
    Generate the injection composition script for a given composition.

    Parameters:
    ----------
    Comp (str): The composition string to use for the injection composition.
    If None, the default composition will be used.

    Returns:
    -------
    str: The generated injection composition script for the given composition.
    """

    if Comp is None:
        Comp = default_comp
    else:
        Comp = Comp
    return Comp


def generate_all_scripts(ng_data_directory='../../data/agsi_depleted/',
                         sch_directory_path='../../Data_scripts/Schedule_'
                         'files',
                         data_directory_path='../../Data_scripts/DATA_Files',
                         inj_composition=Full_H2_comp,
                         time_unit='months', df=None):
    """
    Generate both Data and Schedule scripts for each site in the DataFrame.

    Parameters:
    ----------
    ng_data_directory (str): The directory path where the NG data from AGSI is
    stored.
    sch_directory_path (str): The directory path where the schedule files will
    be stored.
    data_directory_path (str): The directory path where the DATA files will
    be stored.
    inj_composition (str): The injection composition string to use. If None,
    the default composition will be used.

    Returns:
    -------
    None
    """
    if df is None:
        df = df_all_cycles(ng_data_directory, MAXrate=6 *
                           (10**5), time_unit=time_unit)
        print(df)
    for site, site_df in df.groupby('Site'):
        if type(site) is str:
            site = site.replace(' ', '_')
            site = site.upper()
        script = initial_script_named(
            site) + "\n".join(site_df['Row_Script'].tolist())
        with open(f"{sch_directory_path}/{site}_SCH.INC", "w") as f:
            f.write(script)
        DATA_script = generate_DATA_script_title(
            site) + injection_comp(inj_composition) + find_set_schedule(site)
        with open(f"{data_directory_path}/{site}_SCRIPT.DATA", "w") as f:
            f.write(DATA_script)
