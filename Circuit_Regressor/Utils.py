import sympy as sp
from IPython.display import display, Math
import re

# Definimos w como símbolo real y positivo
w = sp.symbols('w', positive=True, real=True)

# Clase que envuelve una expresión simbólica de impedancia
class Impedancia:
    def __init__(self, expr):
        self.expr = sp.sympify(expr)

    def __add__(self, other):
        if isinstance(other, Impedancia):
            return Impedancia(self.expr + other.expr)
        return Impedancia(self.expr + other)

    def __radd__(self, other):
        return Impedancia(other + self.expr)

    def __mul__(self, other):
        if isinstance(other, Impedancia):
            return Impedancia(self.expr * other.expr)
        return Impedancia(self.expr * other)

    def __rmul__(self, other):
        return Impedancia(other * self.expr)

    def __truediv__(self, other):
        if isinstance(other, Impedancia):
            return Impedancia(self.expr / other.expr)
        return Impedancia(self.expr / other)

    def __rtruediv__(self, other):
        return Impedancia(other / self.expr)

    def simplify(self):
        self.expr = sp.together(sp.simplify(self.expr))
        return self

    def as_expr(self):
        return self.expr

    def __repr__(self):
        return str(self.expr)

# Operadores de conexión
def s(z1, z2):
    """Conexión en serie: Z_total = Z1 + Z2"""
    return z1 + z2

def p(z1, z2):
    """Conexión en paralelo: Z_total = Z1*Z2/(Z1+Z2)"""
    return z1 * z2 / (z1 + z2)

def parse_impedancia(cadena):
    """
    Recibe una cadena (por ejemplo, "s(R, p(R, L))") y devuelve la expresión
    simbólica simplificada en la forma A + I*B (en una sola línea),
    asignando parámetros únicos a cada componente.
    """
    # Contadores para componentes (para asignar R1, R2, etc.)
    counters = {'R': 0, 'L': 0, 'C': 0}

    # Funciones locales que crean un nuevo símbolo con asunción real y positivo.
    def R_func():
        counters['R'] += 1
        return Impedancia(sp.symbols(f'R{counters["R"]}', positive=True, real=True))

    def L_func():
        counters['L'] += 1
        # Impedancia del inductor: 1j*w*L_i
        return Impedancia(sp.I * w * sp.symbols(f'L{counters["L"]}', positive=True, real=True))

    def C_func():
        counters['C'] += 1
        # Impedancia del condensador: 1/(1j*w*C_i)
        return Impedancia(1/(sp.I * w * sp.symbols(f'C{counters["C"]}', positive=True, real=True)))

    # Entorno para eval: se definen las funciones y símbolos
    env = {
        'R': R_func,
        'L': L_func,
        'C': C_func,
        's': s,
        'p': p,
        'w': w,
        'I': sp.I,
    }

    # Para que cada aparición de R, L o C sin paréntesis se invoque como función
    patron = r'\b([RLC])\b(?!\()'
    cadena_proc = re.sub(patron, r'\1()', cadena)

    # Se evalúa la cadena y se simplifica la expresión
    imp = eval(cadena_proc, {}, env)
    imp.simplify()
    expr = sp.together(imp.as_expr())

    # Se extraen y simplifican las partes real e imaginaria
    A = sp.factor(sp.re(expr), w)
    B = sp.factor(sp.im(expr), w)
    final_expr = sp.together(A) + sp.I * sp.together(B)

    return final_expr

def display_impedancia(expr):
    """Muestra la expresión simbólica en un display (LaTeX)"""
    sp.init_printing()  # Inicializa impresión “bonita”
    display(Math(sp.latex(expr)))