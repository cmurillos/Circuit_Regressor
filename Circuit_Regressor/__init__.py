"""
Circuit_Regressor - Un paquete para regresión simbólica de circuitos eléctricos

Este paquete permite encontrar estructuras de circuitos eléctricos que coincidan
con una respuesta de impedancia dada, utilizando algoritmos evolutivos.
"""

# Importamos las clases y funciones principales del módulo Circuit_Regressor
from .Circuit_Regressor import (
    CircuitComponent,
    ElectricalSymbolicRegressor,
    symbolic_regressor_circuit,
    w_sym
)

# Importamos las utilidades del módulo Utils
from .Utils import (
    Impedancia,
    s,
    p,
    parse_impedancia,
    display_impedancia,
    w
)

# Definimos la versión del paquete
__version__ = '0.1.0'

# Definimos qué se importa con "from Circuit_Regressor import *"
__all__ = [
    # Clases y funciones principales
    'CircuitComponent',
    'ElectricalSymbolicRegressor',
    'symbolic_regressor_circuit',
    'w_sym',
    
    # Utilidades
    'Impedancia',
    's',
    'p',
    'parse_impedancia',
    'display_impedancia',
    'w'
]

# Mensaje informativo al importar el paquete
print(f"Circuit_Regressor v{__version__} - Regresión simbólica de circuitos eléctricos")

