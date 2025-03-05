import numpy as np
import sympy as sp
import hashlib
import copy
import time
from tqdm import tqdm
from scipy.optimize import differential_evolution
import tensorflow as tf

w_sym = sp.symbols('w', positive=True, real=True)                                                                 # Declaramos el símbolo para la frecuencia (usado en las expresiones simbólicas)

class CircuitComponent:
    def __init__(self, operator, children=None, param_bounds=None, optimized_value=None):
        self.operator = operator
        self.children = children or []
        self.param_bounds = param_bounds
        self.optimized_value = optimized_value                                                                    # Aquí se almacenará la función lambda (compilada a partir de la expresión simbólica),
        self.cached_lambda = None                                                                                 # así como los símbolos y la expresión
        self.cached_symbols = None
        self.cached_expr = None

class ElectricalSymbolicRegressor:
    def __init__(self, num_components=15, iterations_SR=15, R_max=2, L_max=2, C_max=2, R_bound=(1e-5,300), L_bound=(1e-12,1), C_bound=(1e-12, 1)):
        self.num_components = num_components
        self.iterations = iterations_SR
        self.R_max = R_max
        self.L_max = L_max
        self.C_max = C_max
        self.R_bound = R_bound
        self.L_bound = L_bound
        self.C_bound = C_bound
        self.unary_operators = ['R', 'L', 'C']
        self.binary_operators = ['serie', 'paralelo']
        self.history = []
        self.unique_structures = set()

    # ================== FUNCIONES BASE ==================
    def _evaluate_R(self, w, param): return param

    def _evaluate_L(self, w, param): return 1j * w * param

    def _evaluate_C(self, w, param): return 1/(1j * w * param)

    def _evaluate_serie(self, a, b): return a + b

    def _evaluate_paralelo(self, a, b): return (a * b) / (a + b)

    # =============== MANEJO DE INDICES ===============
    def _count_components(self, component):
        counts = {'R': 0, 'L': 0, 'C': 0}
        def traverse(c):
            if c.operator in counts:
                counts[c.operator] += 1
            for child in c.children:
                traverse(child)
        traverse(component)
        return counts

    # =============== MANEJO DE PROBABILIDADES ===============
    def _get_component_probabilities(self, component):
        counts = self._count_components(component)
        available = {
            'R': max(self.R_max - counts['R'], 0),
            'L': max(self.L_max - counts['L'], 0),
            'C': max(self.C_max - counts['C'], 0)
        }
        total = np.sum(list(available.values()))
        if total == 0:
            return {'R': 1/3, 'L': 1/3, 'C': 1/3}
        probs = {k: v/total for k, v in available.items()}
        probs = {k: np.clip(v, 0.05, 0.95) for k, v in probs.items()}
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}

    def _select_unary_operator(self, component):
        probs = self._get_component_probabilities(component)
        probs_list = [probs['R'], probs['L'], probs['C']]
        probs_list = np.array(probs_list) / np.sum(probs_list)
        return np.random.choice(self.unary_operators, p=probs_list)

    # =============== GENERACIÓN DE COMPONENTES ===============
    def _generate_unique_components(self):
        components = []
        unique_hashes = set()
        with tqdm(total=self.num_components, desc="Generando componentes únicos") as pbar:
            while len(components) < self.num_components:
                new_comp = self._generate_valid_component()
                comp_hash = self._component_hash(new_comp)
                if comp_hash not in unique_hashes:
                    components.append(new_comp)
                    unique_hashes.add(comp_hash)
                    pbar.update(1)
                if pbar.n > self.num_components * 15:
                    raise RuntimeError("No se pueden generar componentes únicos suficientes")
        return components

    def _component_hash(self, component):
        structure = self._component_to_structure(component)
        return hashlib.md5(structure.encode()).hexdigest()

    def _generate_valid_component(self, depth=0):
        for _ in range(100):
            component = self._generate_random_component(depth)
            if self._is_valid_component(component):
                return component
        raise RuntimeError("No se pudo generar componente válido")

    def _generate_random_component(self, depth=0):
        if depth >= 3 or np.random.rand() < 0.4:
            op = self._select_unary_operator(CircuitComponent('temp'))
            if op == 'R': bounds = self.R_bound  
            elif op=='L': bounds = self.L_bound 
            elif op=='C': bounds = self.C_bound
            return CircuitComponent(op, param_bounds=bounds)
        op = np.random.choice(self.binary_operators)
        return CircuitComponent(
            op,
            children=[
                self._generate_random_component(depth + 1),
                self._generate_random_component(depth + 1)
            ]
        )

    # =============== MODIFICACIÓN DE COMPONENTES ===============
    def _modify_component(self, original_component):
        component = copy.deepcopy(original_component)
        action = np.random.choice(["grow", "prune", "change"], p=[0.5, 0.3, 0.2])
        if action == "grow":
            terminals = self._find_terminals(component)
            if terminals:
                target = np.random.choice(terminals)
                target.operator = np.random.choice(self.binary_operators)
                new_op_left = self._select_unary_operator(component)
                new_op_right = self._select_unary_operator(component)
                target.children = [
                    CircuitComponent(
                        new_op_left,
                        param_bounds=(0, 300) if new_op_left == 'R' else (1e-12, 1)
                    ),
                    CircuitComponent(
                        new_op_right,
                        param_bounds=(0, 300) if new_op_right == 'R' else (1e-12, 1)
                    )
                ]
        elif action == "prune":
            binaries = self._find_binary_nodes(component)
            if binaries:
                target = np.random.choice(binaries)
                new_op = self._select_unary_operator(component)
                target.operator = new_op
                target.children = []
                target.param_bounds = (0, 300) if new_op == 'R' else (1e-12, 1)
        elif action == "change":
            nodes = self._traverse_component(component)
            if nodes:
                target = np.random.choice(nodes)
                if target.operator in self.unary_operators:
                    new_op = self._select_unary_operator(component)
                    target.operator = new_op
                    target.param_bounds = (0, 300) if new_op == 'R' else (1e-12, 1)
        component.cached_lambda = None                                                                          # Invalida la caché, ya que la estructura pudo haber cambiado.
        component.cached_symbols = None
        component.cached_expr = None
        return component if self._is_valid_component(component) else original_component

    def _find_terminals(self, component):
        terminals = []
        if not component.children:
            terminals.append(component)
        for child in component.children:
            terminals += self._find_terminals(child)
        return terminals

    def _find_binary_nodes(self, component):
        binaries = []
        if component.operator in self.binary_operators:
            binaries.append(component)
        for child in component.children:
            binaries += self._find_binary_nodes(child)
        return binaries

    def _traverse_component(self, component):
        nodes = [component]
        for child in component.children:
            nodes += self._traverse_component(child)
        return nodes

    # =============== VALIDACIÓN ===============
    def _has_invalid_combinations(self, component):
        invalid = False
        def traverse(c):
            nonlocal invalid
            if c.operator in self.binary_operators:
                if len(c.children) == 2 and c.children[0].operator == c.children[1].operator:
                    invalid = True
            for child in c.children:
                traverse(child)
        traverse(component)
        return invalid

    def _has_invalid_series(self, component):
        invalid = False
        current_series_components = set()
        def traverse(node, in_series=False):
            nonlocal invalid, current_series_components
            if node.operator == 'serie':
                local_components = set()
                for child in node.children:
                    if child.operator in self.unary_operators:
                        if child.operator in local_components:
                            invalid = True
                        local_components.add(child.operator)
                    traverse(child, in_series=True)
                current_series_components.update(local_components)
            elif in_series and node.operator in self.unary_operators:
                if node.operator in current_series_components:
                    invalid = True
                current_series_components.add(node.operator)
            else:
                for child in node.children:
                    traverse(child, in_series=False)
        traverse(component)
        return invalid

    def _is_valid_component(self, component):
        counts = self._count_components(component)
        return (counts['R'] <= self.R_max and
                counts['L'] <= self.L_max and
                counts['C'] <= self.C_max and
                not self._has_invalid_combinations(component) and
                not self._has_invalid_series(component))

    def _ensure_unique_component(self, component, unique_structures, max_attempts=5):
        def canonicalize(node):
            if not node.children:
                return
            for child in node.children:
                canonicalize(child)
            if node.operator in ['serie', 'paralelo']:
                node.children.sort(key=lambda x: self._component_to_structure(x))
        def canonical(node):
            if not node.children:
                return node.operator
            if node.operator in ['serie', 'paralelo']:
                children = [canonical(child) for child in node.children]
                children.sort()
                if node.operator == 'serie':
                    return "s(" + ",".join(children) + ")"
                else:
                    return "p(" + ",".join(children) + ")"
            else:
                return node.operator + "(" + ",".join(canonical(child) for child in node.children) + ")"
        canonicalize(component)
        canon = canonical(component)
        comp_hash = hashlib.md5(canon.encode()).hexdigest()
        attempts = 0
        new_component = component
        while comp_hash in unique_structures and attempts < max_attempts:
            new_component = self._modify_component(new_component)
            canonicalize(new_component)
            canon = canonical(new_component)
            comp_hash = hashlib.md5(canon.encode()).hexdigest()
            attempts += 1
        return new_component, comp_hash

    # =============== EVALUACIÓN Y OPTIMIZACIÓN ===============
    def _component_to_structure(self, component):
        if component.operator in self.unary_operators:
            return component.operator[0].upper()
        elif component.operator == 'serie':
            return f"s({', '.join(self._component_to_structure(c) for c in component.children)})"
        else:
            return f"p({', '.join(self._component_to_structure(c) for c in component.children)})"

    def _get_parameters_list(self, component):
        params = []
        def traverse(c):
            if c.operator in self.unary_operators and c.optimized_value is not None:
                params.append(c.optimized_value)
            for child in c.children:
                traverse(child)
        traverse(component)
        return params

    def _get_parameters(self, component):
        params, bounds = [], []
        def traverse(c):
            if c.operator in self.unary_operators:
                params.append(c.optimized_value if c.optimized_value is not None else np.mean(c.param_bounds))
                bounds.append(c.param_bounds)
            for child in c.children:
                traverse(child)
        traverse(component)
        return params, bounds

    def _update_parameters(self, component, values):
        idx = 0
        def traverse(c):
            nonlocal idx
            if c.operator in self.unary_operators:
                c.optimized_value = values[idx]
                idx += 1
            for child in c.children:
                traverse(child)
        traverse(component)

    # --- NUEVA METODOLOGÍA: Construcción y compilación simbólica de la impedancia ---
    def _build_symbolic_expression_rec(self, component, symbols_iter):
        if component.operator in self.unary_operators:
            p = next(symbols_iter)
            if component.operator == 'R':
                return p
            elif component.operator == 'L':
                return sp.I * w_sym * p
            elif component.operator == 'C':
                return 1/(sp.I * w_sym * p)
        elif component.operator == 'serie':
            return sp.Add(*[self._build_symbolic_expression_rec(c, symbols_iter) for c in component.children])
        elif component.operator == 'paralelo':
            exprs = [self._build_symbolic_expression_rec(c, symbols_iter) for c in component.children]
            return sp.Mul(*exprs) / sp.Add(*exprs)

    def _build_symbolic_expression(self, component):                                                                  # Contamos cuántos nodos unarios (R, L, C) hay en el árbol.
        count = 0
        def count_unary(c):
            nonlocal count
            if c.operator in self.unary_operators:
                count += 1
            for child in c.children:
                count_unary(child)
        count_unary(component)
        symbols = sp.symbols('p0:%d' % count, positive=True, real=True)
        symbols_iter = iter(symbols)
        expr = self._build_symbolic_expression_rec(component, symbols_iter)                                           # Intentamos factorizar la parte real e imaginaria con respecto a w_sym.
        try:
            A = sp.factor(sp.re(expr), w_sym)
        except Exception as e:
            A = sp.collect(sp.together(sp.re(expr)), w_sym)
        try:
            B = sp.factor(sp.im(expr), w_sym)
        except Exception as e:
            B = sp.collect(sp.together(sp.im(expr)), w_sym)
        expr = sp.together(A + sp.I * B)
        return expr, symbols

    def _evaluate_component(self, component, w, Z):
        params, bounds = self._get_parameters(component)
        if not params:
            return float('inf'), []
        if component.cached_lambda is None:
            expr, symbols = self._build_symbolic_expression(component)
            f = sp.lambdify((symbols, w_sym), expr, 'numpy')
            component.cached_lambda = f
            component.cached_symbols = symbols
            component.cached_expr = expr
        f = component.cached_lambda
        def cost(x):
            x = np.array(x).ravel()
            if x.size != len(component.cached_symbols):
                raise ValueError(f"Se esperaban {len(component.cached_symbols)} parámetros, pero se recibió {x.size}.")
            Z_pred = f(x, w)
            error_norma = (np.real(Z)-np.real(Z_pred))**2 + (np.imag(Z)-np.imag(Z_pred))**2
            error_norma = np.mean(error_norma)
            return error_norma
        result = differential_evolution(cost,
                                        bounds,
                                        seed=None,
                                        popsize=50,
                                        maxiter=1000,
                                        mutation=(0.5, 1.5),
                                        recombination=0.7,
                                        strategy='best1bin',
                                        init='sobol',
                                        updating='deferred',
                                        tol=1e-14,
                                        vectorized=False)
        optimized_params = result.x.tolist()
        self._update_parameters(component, optimized_params)
        best_params = result.x
        best_score = result.fun
        return best_score, optimized_params

    # ============ Principal function ================

    def fit(self, w, Z):
        components = self._generate_unique_components()
        historical_errors = []
        unique_structures = set()
        with tqdm(total=self.iterations, desc="Optimizando modelos") as pbar:
            for iteration in range(self.iterations):
                new_components = []
                current_errors = []
                for comp in components:
                    new_comp = self._modify_component(comp)
                    if not self._is_valid_component(new_comp):
                        new_components.append(comp)
                        continue
                    new_comp, structure_hash = self._ensure_unique_component(new_comp, unique_structures)
                    error_old, params_old = self._evaluate_component(comp, w, Z)
                    error_new, params_new = self._evaluate_component(new_comp, w, Z)
                    current_errors.extend([error_old, error_new])
                    diversity = np.std(historical_errors[-20:]) if historical_errors else 1.0
                    T = max(0.1, 1.0 - (iteration / self.iterations)) * (1.0 + 0.5 * (1 - diversity))
                    with np.errstate(over='ignore'): accept_prob = np.exp(-(error_new - error_old) / (T + 0.001))
                    if error_new < error_old or np.random.rand() < accept_prob:
                        new_components.append(new_comp)
                        self.history.append((self._component_to_structure(new_comp), params_new, error_new))
                        unique_structures.add(structure_hash)
                    else:
                        new_components.append(comp)
                        comp_hash = self._component_hash(comp)
                        if comp_hash not in unique_structures:
                            self.history.append((self._component_to_structure(comp), params_old, error_old))
                            unique_structures.add(comp_hash)
                    pbar.set_postfix({'Min Error Iter': f"{np.sqrt(min(h[2] for h in self.history if h[2] < float('inf'))):.3f}"})
                historical_errors.append(np.mean(current_errors))
                components = new_components
                pbar.update(1)
        seen = set()
        unique_history = []
        for h in self.history:
            h_hash = hashlib.md5(h[0].encode()).hexdigest()
            if h_hash not in seen:
                seen.add(h_hash)
                unique_history.append(h)
        self.history = sorted([h for h in unique_history if h[2] < float('inf')], key=lambda x: x[2])
        return self

def symbolic_regressor_circuit(w, Z, num_components=20, iterations=10, R_max=2, L_max=2, C_max=2, R_bound=(1e-5,300), L_bound=(1e-12,1), C_bound=(1e-12, 1)):
    esr = ElectricalSymbolicRegressor(num_components, iterations, R_max, L_max, C_max, R_bound, L_bound, C_bound)
    w, Z = np.array(w), np.array(Z)
    esr.fit(w, Z)
    print("\nMejores modelos:")
    for i, (model, params, error) in enumerate(esr.history):
        print(f"\nModelo {i+1}:")
        print(f"Estructura: {model}")
        print(f"Parámetros: {np.round(params, 5)}")
        print(f"Error norma: {np.sqrt(error):.5f}")
