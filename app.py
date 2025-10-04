import os
import sqlite3

from flask import Flask, flash, redirect, render_template, request, session, jsonify
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np
import pulp
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpInteger

app = Flask(__name__)

# Production configuration
import os
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

DATABASE = "inventory.db"

def get_db_connection():
    """Get a database connection with row factory for dict-like access"""
    # Use absolute path for production
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATABASE)
    print(f"Attempting to connect to database at: {db_path}", flush=True)
    print(f"Database file exists: {os.path.exists(db_path)}", flush=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def execute_query(query, *params):
    """Execute a query and return results as list of dicts"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        
        # For SELECT queries, fetch results
        if query.strip().upper().startswith('SELECT'):
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        else:
            # For INSERT, UPDATE, DELETE
            conn.commit()
            return cursor.rowcount
    except Exception as e:
        print(f"Database error: {e}")
        raise e
    finally:
        conn.close()

def execute_transaction(operations):
    """Execute multiple operations in a single transaction"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        for query, params in operations:
            cursor.execute(query, params)
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Transaction error: {e}")
        raise e
    finally:
        conn.close()

def init_db():
    """Initialize database connection and create tables if they don't exist"""
    conn = get_db_connection()
    try:
        # Check if database file exists and is accessible
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Database initialized. Found {len(tables)} tables.")
        return True
    except Exception as e:
        print(f"Database initialization error: {e}")
        return False
    finally:
        conn.close()


@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
def index():
    try:
        # Test database connection
        test_connection = get_db_connection()
        test_connection.close()
        print("Database connection successful", flush=True)
        
        # Try to render template
        return render_template("layout.html")
    except Exception as e:
        print(f"Error in index route: {e}", flush=True)
        print(f"Error type: {type(e).__name__}", flush=True)
        
        # Check if it's a template error
        if "layout.html" in str(e) or "TemplateNotFound" in str(e):
            return f"Template error: Missing layout.html template file. Error: {str(e)}", 500
        else:
            return f"Database connection error: {str(e)}", 500


@app.route("/portafolio", methods=["GET", "POST"])
def portafolio():
    products = execute_query("SELECT * FROM products")
    return render_template("portafolio.html", products=products)


@app.route("/inventory", methods=["GET", "POST"])
def inventory():
    stock = execute_query(
        "SELECT p.SKU,p.NAME,i.EXPIRED,i.DAY,i.VOLUME,p.EXPIRATION FROM inventory AS i JOIN products AS p ON i.SKU = p.SKU;")
    return render_template("inventory.html", stock=stock)


@app.route("/simulate", methods=["GET"])
def simulation():
    """Renders the simulation page."""
    return render_template("simulate.html")


@app.route("/run_simulation", methods=["POST"])
def run_simulation():
    """Runs the simulation and returns JSON results."""

    try:
        # Clear previous simulation data
        print("Clearing previous simulation data...", flush=True)
        execute_query("DELETE FROM Simulation")
        print("Previous simulation data cleared.", flush=True)

        # Get and validate input data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        size = data.get('size', 30)
        distribution = data.get('lead_time_distribution', 'normal')
        param1 = data.get('lead_time_param1', 7.0)
        param2 = data.get('lead_time_param2', 2.0)

        # Validate and convert inputs
        try:
            size = int(size)
            param1 = float(param1)
            param2 = float(param2)

            if size <= 0 or size > 1000:
                return jsonify({"error": "Size must be between 1 and 1000"}), 400

            if distribution not in ['normal', 'uniform']:
                return jsonify({"error": "Distribution must be 'normal' or 'uniform'"}), 400

        except (ValueError, TypeError):
            return jsonify({"error": "Invalid parameter values"}), 400

        # Generate lead times
        rng = np.random.default_rng()

        

        # Get all SKUs
        sim_parameters = execute_query("SELECT DISTINCT SKU FROM transactions")

        if not sim_parameters:
            return jsonify({"error": "No SKUs found in transactions table"}), 404

        # Process each SKU
        results = []
        simulation_inserts = []

        # Process simulations
        try:
            NUM_ITERATIONS = 10
            all_simulation_inserts = []

            for iteration in range(1, NUM_ITERATIONS + 1):
                simulation_inserts = []
                

                for sku_row in sim_parameters:
                    sku = sku_row["SKU"]

                    if distribution == 'normal':
                        lead_times = rng.normal(param1, param2, size)
                        lead_times = np.maximum(lead_times, 0.1)  # Ensure positive
                    else:  # uniform
                        lead_times = rng.uniform(param1, param2, size)

                    lead_times = np.round(lead_times).astype(int).tolist()    

                    # Get product information including margin
                    product_info = execute_query("SELECT SKU, MARGIN, NAME FROM products WHERE SKU = ?", sku)
                    margin = float(product_info[0]["MARGIN"]) if product_info and product_info[0]["MARGIN"] else 0.0
                    product_name = product_info[0]["NAME"] if product_info else "Unknown"

                    print(f"SKU {sku} ({product_name}): Margin = {margin}", flush=True)

                    # Get initial inventory for this SKU
                    initial_inventory = execute_query(
                        "SELECT SKU, SUM(VOLUME) as STOCK FROM inventory WHERE SKU = ?", sku)

                    if initial_inventory and initial_inventory[0]["STOCK"] is not None:
                        initial_stock = float(initial_inventory[0]["STOCK"])
                    else:
                        initial_stock = 0.0

                    print(f"SKU {sku}: Initial stock = {initial_stock}", flush=True)

                    # Get volumes for this SKU (for demand calculation)
                    sku_data = execute_query("SELECT VOLUME FROM transactions WHERE SKU = ?", sku)
                    volumes = [float(row["VOLUME"]) for row in sku_data]

                    if len(volumes) < 2:
                        continue  # Skip SKUs with insufficient data

                    # Calculate demand statistics
                    avg_vol = sum(volumes) / len(volumes)
                    variance = sum((v - avg_vol) ** 2 for v in volumes) / (len(volumes) - 1)
                    sd_vol = variance ** 0.5 if variance > 0 else avg_vol * 0.1

                    # Generate demand values
                    demand_array = rng.normal(loc=avg_vol, scale=sd_vol, size=size)
                    demand_array = np.maximum(demand_array, 0)
                    demand_integers = np.round(demand_array).astype(int).tolist()

                    # Calculate running stock levels
                    current_stock = initial_stock
                    stock_levels = []

                    for daily_demand in demand_integers:
                        stock_levels.append(round(current_stock, 2))  # Save BEFORE subtracting demand
                        current_stock -= daily_demand

                    # Calculate total unfulfilled demand and lost profit for statistics
                    total_unfulfilled = 0
                    total_lost_profit = 0.0

                    # Add to results (with extra safety checks)
                    results.append({
                        "SKU": str(sku),
                        "product_name": product_name,
                        "margin": round(float(margin), 2),
                        "initial_stock": round(float(initial_stock), 2),
                        "a_volume": round(float(avg_vol), 2),
                        "sd_volume": round(float(sd_vol), 2),
                        "generated_values": demand_integers,
                        "lead_time_values": lead_times,
                        "stock_levels": stock_levels,
                        "final_stock": round(float(stock_levels[-1]), 2) if stock_levels else round(float(initial_stock), 2),
                        "stats": {
                            "min_demand": int(min(demand_integers)) if demand_integers else 0,
                            "max_demand": int(max(demand_integers)) if demand_integers else 0,
                            "mean_demand": round(float(np.mean(demand_integers)), 1) if demand_integers else 0.0,
                            "min_stock": round(float(min(stock_levels)), 2) if stock_levels else round(float(initial_stock), 2),
                            "max_stock": round(float(max(stock_levels)), 2) if stock_levels else round(float(initial_stock), 2),
                            "days_negative": sum(1 for stock in stock_levels if stock < 0) if stock_levels else 0
                        }
                    })

                    # Prepare simulation data for database insertion
                    unmet_demand_carryover = 0

                    for j, demand_value in enumerate(demand_integers):
                        lead_time_value = lead_times[j] if j < len(lead_times) else lead_times[-1]
                        stock_before_demand = stock_levels[j] if j < len(stock_levels) else initial_stock
                        stock_after_demand = stock_before_demand - demand_value

                        # If stock is negative, accumulate unmet demand
                        if stock_after_demand < 0:
                            unmet_demand_carryover += abs(stock_after_demand) if stock_before_demand >= 0 else demand_value
                            not_billed = unmet_demand_carryover
                        else:
                            unmet_demand_carryover = 0
                            not_billed = 0

                        total_unfulfilled += not_billed
                        lost_profit = not_billed * margin
                        total_lost_profit += lost_profit

                        simulation_inserts.append((
                            str(sku),
                            stock_before_demand,
                            demand_value,
                            None,
                            None,
                            lead_time_value,
                            not_billed,
                            margin * not_billed,
                            iteration
                        ))

                # Add simulation data for batch insert
                if simulation_inserts:
                    all_simulation_inserts.extend(simulation_inserts)
            
            # Perform batch insert using transaction
            if all_simulation_inserts:
                operations = []
                for insert_data in all_simulation_inserts:
                    operations.append(("""
                        INSERT INTO Simulation
                        (SKU, STOCK, DEMAND, SHIPMENT_MADE, Stock_Received, LEAD_TIME, NOT_BILLED, COO, Iteration)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, insert_data))
                execute_transaction(operations)

            print(f"Simulation completed. Inserted {len(simulation_inserts)} records.", flush=True)

            # --- NUEVO BLOQUE PARA AGREGAR ESTADÍSTICAS DE LAS 10 ITERACIONES ---
            from collections import defaultdict
            sku_stats = defaultdict(lambda: {
                "SKU": None,
                "product_name": None,
                "margin": None,
                "initial_stock": [],
                "a_volume": [],
                "sd_volume": [],
                "generated_values": [],
                "lead_time_values": [],
                "stock_levels": [],
                "final_stock": [],
                "min_demand": [],
                "max_demand": [],
                "mean_demand": [],
                "min_stock": [],
                "max_stock": [],
                "days_negative": []
            })

            for r in results:
                sku = r["SKU"]
                sku_stats[sku]["SKU"] = sku
                sku_stats[sku]["product_name"] = r["product_name"]
                sku_stats[sku]["margin"] = r["margin"]
                sku_stats[sku]["initial_stock"].append(r["initial_stock"])
                sku_stats[sku]["a_volume"].append(r["a_volume"])
                sku_stats[sku]["sd_volume"].append(r["sd_volume"])
                sku_stats[sku]["generated_values"].append(r["generated_values"])
                sku_stats[sku]["lead_time_values"].append(r["lead_time_values"])
                sku_stats[sku]["stock_levels"].append(r["stock_levels"])
                sku_stats[sku]["final_stock"].append(r["final_stock"])
                sku_stats[sku]["min_demand"].append(r["stats"]["min_demand"])
                sku_stats[sku]["max_demand"].append(r["stats"]["max_demand"])
                sku_stats[sku]["mean_demand"].append(r["stats"]["mean_demand"])
                sku_stats[sku]["min_stock"].append(r["stats"]["min_stock"])
                sku_stats[sku]["max_stock"].append(r["stats"]["max_stock"])
                sku_stats[sku]["days_negative"].append(r["stats"]["days_negative"])

            # Calcula estadísticas agregadas por SKU
            aggregated_results = []
            for sku, data in sku_stats.items():
                # Unifica todos los valores generados y stock_levels de las 10 simulaciones para estadísticas globales
                all_demands = [d for sublist in data["generated_values"] for d in sublist]
                all_stocks = [s for sublist in data["stock_levels"] for s in sublist]

                aggregated_results.append({
                    "SKU": sku,
                    "product_name": data["product_name"],
                    "margin": data["margin"],
                    "initial_stock_avg": np.mean(data["initial_stock"]),
                    "a_volume_avg": np.mean(data["a_volume"]),
                    "sd_volume_avg": np.mean(data["sd_volume"]),
                    "simulation_days": size,
                    "lead_time_distribution": distribution,
                    "lead_time_param1": param1,
                    "lead_time_param2": param2,
                    # Estadísticas globales sobre los 10 runs
                    "min_demand": int(np.min(all_demands)) if all_demands else 0,
                    "avg_demand": float(np.mean(all_demands)) if all_demands else 0,
                    "max_demand": int(np.max(all_demands)) if all_demands else 0,
                    "min_stock": float(np.min(all_stocks)) if all_stocks else 0,
                    "max_stock": float(np.max(all_stocks)) if all_stocks else 0,
                    "days_negative_avg": float(np.mean(data["days_negative"])) if data["days_negative"] else 0,
                    # Estadísticas por simulación (promedios de los 10 runs)
                    "mean_demand_per_run": float(np.mean(data["mean_demand"])) if data["mean_demand"] else 0,
                    "min_demand_per_run": float(np.mean(data["min_demand"])) if data["min_demand"] else 0,
                    "max_demand_per_run": float(np.mean(data["max_demand"])) if data["max_demand"] else 0,
                    "min_stock_per_run": float(np.mean(data["min_stock"])) if data["min_stock"] else 0,
                    "max_stock_per_run": float(np.mean(data["max_stock"])) if data["max_stock"] else 0,
                })

            return jsonify({
                "status": "success",
                "message": f"Generated simulation data for {len(aggregated_results)} SKUs (aggregated over {NUM_ITERATIONS} iterations)",
                "total_records": len(aggregated_results),
                "simulation_size": size,
                "lead_time_distribution": distribution,
                "results": aggregated_results
            })

        except Exception as e:
            print("Simulation error (inside transaction):", e, flush=True)
            raise e

    except Exception as e:
        print("Simulation error:", e, flush=True)
        return jsonify({
            "error": "Internal server error during simulation",
            "details": str(e)
        }), 500

@app.route("/transactions", methods=["GET", "POST"])
def transactions():
    trades = execute_query("SELECT * FROM transactions;")
    return render_template("transactions.html", trades=trades)

@app.route("/optimize", methods=["GET", "POST"])
def optimization():
    # Para peticiones GET, simplemente muestra la página con el botón
    if request.method == "GET":
        # Check if simulation data exists
        try:
            sim_count = execute_query("SELECT COUNT(*) as count FROM Simulation")
            print(f"GET /optimize: Found {sim_count[0]['count']} simulation records", flush=True)
            if sim_count[0]['count'] == 0:
                return render_template("optimize.html", error="No simulation data found. Please run a simulation first.")
            else:
                # Show some info about available data
                skus_with_data = execute_query("SELECT DISTINCT SKU FROM Simulation")
                iterations = execute_query("SELECT DISTINCT Iteration FROM Simulation ORDER BY Iteration DESC LIMIT 3")
                print(f"GET /optimize: Available SKUs: {[s['SKU'] for s in skus_with_data]}, Recent iterations: {[i['Iteration'] for i in iterations]}", flush=True)
        except Exception as e:
            print(f"GET /optimize: Database error: {e}", flush=True)
            return render_template("optimize.html", error=f"Database error: {str(e)}")
        
        return render_template("optimize.html")
    
    # Solo ejecuta la optimización en peticiones POST (cuando se hace clic en el botón)
    try:
        print("POST /optimize: Starting optimization process...", flush=True)
        # Parámetros críticos para el balance correcto
        shipping_cost = 100  # Costo fijo por envío
        holding_cost_rate = 0.01  # 1% del valor del producto por día de almacenamiento
        stockout_penalty_multiplier = 2.0  # Multiplicador sobre el margen para penalizar stockouts

        # Get product info
        products = execute_query("SELECT SKU, NAME, MARGIN, COST FROM products")
        product_info = {prod['SKU']: prod for prod in products}

        # Get all SKUs
        all_skus = [prod['SKU'] for prod in products]

        # Prepare results structure
        all_optimizations = {}

        for sku in all_skus:
            all_optimizations[sku] = []
            for iteration in range(1, 11):
                # Get simulation data for this SKU and iteration
                sim_rows = execute_query(
                    "SELECT * FROM Simulation WHERE SKU = ? AND Iteration = ? ORDER BY Count",
                    sku, iteration
                )
                if not sim_rows:
                    print(f"No simulation data found for SKU {sku}, iteration {iteration}", flush=True)
                    continue

                demands = [int(row['DEMAND']) for row in sim_rows]
                lead_times = [int(row['LEAD_TIME']) for row in sim_rows]
                initial_stock = float(sim_rows[0]['STOCK']) + int(sim_rows[0]['DEMAND']) if sim_rows else 0

                unit_cost = float(product_info[sku]['COST']) if product_info[sku]['COST'] else 10
                margin = float(product_info[sku]['MARGIN']) if product_info[sku]['MARGIN'] else 0

                # Run optimization for this iteration
                result = optimize_purchase_schedule(
                    sku=sku,
                    demands=demands,
                    lead_times=lead_times,
                    initial_stock=initial_stock,
                    unit_cost=unit_cost,
                    margin=margin,
                    shipping_cost=shipping_cost,
                    holding_cost_rate=holding_cost_rate,
                    stockout_penalty_multiplier=stockout_penalty_multiplier,
                    simulation_days=len(demands)
                )
                # Store result for this iteration
                all_optimizations[sku].append({
                    "iteration": iteration,
                    "result": result
                })

        # Check if we have any optimization results
        total_optimizations = sum(len(runs) for runs in all_optimizations.values())
        if total_optimizations == 0:
            return render_template("optimize.html", error="No optimization results generated. This could mean no simulation data was found or all simulations failed. Please run a new simulation first.")

        # Calcular métricas para el resumen
        total_service_level = 0
        total_iterations = 0
        total_savings = 0

        for runs in all_optimizations.values():
            for run in runs:
                total_service_level += run['result']['optimized_strategy']['service_level']
                total_iterations += 1
                # Estimación simplificada de ahorros (10% del costo de compra)
                total_savings += run['result']['optimized_strategy']['cost_breakdown']['purchase_cost'] * 0.1

        # Crear el resumen
        summary = {
            "total_skus": len([sku for sku, runs in all_optimizations.items() if runs]),
            "simulation_days": len(next(iter(all_optimizations.values()))[0]['result']['optimized_strategy']['purchase_schedule']) if all_optimizations else 0,
            "optimized_total_cost": sum(
                run['result']['optimized_strategy']['total_cost']
                for runs in all_optimizations.values() for run in runs
            ),
            "avg_service_level": total_service_level / total_iterations if total_iterations > 0 else 0,
            "total_savings": total_savings
        }

        return render_template("optimize.html", all_optimizations=all_optimizations, summary=summary)

    except Exception as e:
        print(f"Optimization error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return render_template("optimize.html", error=f"Error optimizing purchase schedule: {str(e)}")


def optimize_purchase_schedule(sku, demands, lead_times, initial_stock, unit_cost, margin,
                             shipping_cost, holding_cost_rate, stockout_penalty_multiplier, simulation_days):
    """
    Optimiza cuando y cuánto pedir para un SKU usando Programación Lineal Entera (ILP).
    Retorna el plan de compras óptimo, minimizando costos totales.
    """
    # Crear problema de optimización
    problem = LpProblem(f"Inventory_Optimization_{sku}", LpMinimize)
    
    # Parámetros
    periods = range(simulation_days)
    max_lead_time = max(lead_times) if lead_times else 5
    
    # Valor de penalización por stockout - usar el margen como costo de oportunidad
    stockout_cost = margin * stockout_penalty_multiplier
    
    # Variables de decisión
    # x[t]: Cantidad a ordenar en el período t
    x = {t: LpVariable(f"order_{t}", lowBound=0, cat=LpInteger) for t in periods}
    
    # i[t]: Nivel de inventario al final del período t
    i = {t: LpVariable(f"inventory_{t}", lowBound=0) for t in periods}
    
    # s[t]: Demanda no satisfecha en período t (shortage/stockout)
    s = {t: LpVariable(f"shortage_{t}", lowBound=0) for t in periods}
    
    # y[t]: Variable binaria, 1 si se hace un pedido en período t, 0 si no
    y = {t: LpVariable(f"order_placed_{t}", cat='Binary') for t in periods}
    
    # Función objetivo: minimizar costo total
    total_cost = (
        # Costo de compra
        lpSum(x[t] * unit_cost for t in periods) +
        # Costo fijo por ordenar
        lpSum(y[t] * shipping_cost for t in periods) +
        # Costo de almacenamiento
        lpSum(i[t] * unit_cost * holding_cost_rate for t in periods) +
        # Penalización por demanda insatisfecha (usar margen como costo de oportunidad)
        lpSum(s[t] * stockout_cost for t in periods)
    )
    problem += total_cost
    
    # Restricciones
    
    # Balance de inventario
    for t in periods:
        # Órdenes que llegan en el período t
        incoming_orders = lpSum(x[t_prime] for t_prime in periods 
                              if t_prime + lead_times[min(t_prime, len(lead_times)-1)] == t)
        
        if t == 0:
            # Balance inicial
            problem += i[t] == initial_stock + incoming_orders - demands[t] + s[t]
        else:
            # Balance en períodos subsecuentes
            problem += i[t] == i[t-1] + incoming_orders - demands[t] + s[t]
    
    # Restricción de tamaño mínimo de orden (100 unidades)
    min_order_size = 100
    for t in periods:
        problem += x[t] >= min_order_size * y[t]
        problem += x[t] <= 10000 * y[t]  # Límite superior arbitrariamente grande
    
    # Restricción de nivel de servicio mínimo (al menos 85%)
    total_demand = sum(demands)
    total_shortage = lpSum(s[t] for t in periods)
    problem += total_shortage <= 0.15 * total_demand  # Máximo 15% de demanda insatisfecha
    
    # Resolver el problema con límite de tiempo y gap de optimalidad
    solver = pulp.PULP_CBC_CMD(timeLimit=45, gapRel=0.05)  # 45 segundos máximo, 5% gap
    problem.solve(solver)
    
    # Verificar si se encontró una solución
    if problem.status != 1:  # 1 = Optimal
        print(f"Warning: Optimization for SKU {sku} didn't reach optimal solution. Status: {problem.status}")
    
    # Extraer resultados
    purchase_schedule = []
    for t in periods:
        purchase_schedule.append({
            'day': t,
            'quantity': int(x[t].value()) if x[t].value() is not None else 0,
            'delivery_day': t + lead_times[min(t, len(lead_times)-1)] if x[t].value() > 0 else None
        })
    
    # Analizar rendimiento de la estrategia optimizada
    stockout_days = sum(1 for t in periods if s[t].value() > 0)
    service_level = 100 * (1 - stockout_days / len(periods))
    
    # Calcular costos detallados
    ordering_cost = sum(shipping_cost * y[t].value() for t in periods)
    purchase_cost = sum(unit_cost * x[t].value() for t in periods)
    holding_cost = sum(unit_cost * holding_cost_rate * i[t].value() for t in periods)
    stockout_cost_total = sum(stockout_cost * s[t].value() for t in periods)
    
    return {
        'sku': sku,
        'product_name': 'Product',  # Puedes obtener este dato de la base de datos
        'optimized_strategy': {
            'purchase_schedule': purchase_schedule,
            'total_cost': ordering_cost + purchase_cost + holding_cost + stockout_cost_total,
            'service_level': service_level,
            'total_orders': sum(1 for day in purchase_schedule if day['quantity'] > 0),
            'total_quantity_ordered': sum(day['quantity'] for day in purchase_schedule),
            'average_inventory': sum(i[t].value() for t in periods) / len(periods),
            'stockout_days': stockout_days,
            'cost_breakdown': {
                'ordering_cost': ordering_cost,
                'purchase_cost': purchase_cost,
                'holding_cost': holding_cost,
                'stockout_cost': stockout_cost_total
            }
        }
    }


def generate_reorder_point_strategy(demands, lead_times, initial_stock, avg_demand, avg_lead_time):
    """Generate reorder point strategy: order when stock falls below threshold."""
    reorder_point = int(avg_demand * avg_lead_time * 1.5)  # 50% safety buffer, rounded to int
    order_quantity = int(avg_demand * 14)  # 2 weeks supply, rounded to int

    schedule = []
    current_stock = initial_stock

    for day in range(len(demands)):
        # Check if we need to place an order
        if current_stock <= reorder_point:
            schedule.append({
                'day': day,
                'order_quantity': order_quantity,
                'delivery_day': min(day + int(lead_times[day]), len(demands) - 1),
                'quantity': order_quantity  # Already int
            })
        else:
            schedule.append({
                'day': day,
                'order_quantity': 0,
                'delivery_day': None,
                'quantity': 0
            })

        # Update stock level
        current_stock -= demands[day]
        current_stock = max(0, current_stock)  # Can't go negative

    return schedule


def generate_fixed_interval_strategy(demands, simulation_days, avg_demand, interval_days):
    """Generate fixed interval strategy: order every N days."""
    schedule = []
    order_quantity = int(avg_demand * interval_days * 1.2)  # 20% buffer, rounded to int

    for day in range(simulation_days):
        if day % interval_days == 0:  # Order every interval_days
            schedule.append({
                'day': day,
                'order_quantity': order_quantity,
                'delivery_day': min(day + 5, simulation_days - 1),  # Assume 5-day lead time
                'quantity': order_quantity  # Already int
            })
        else:
            schedule.append({
                'day': day,
                'order_quantity': 0,
                'delivery_day': None,
                'quantity': 0
            })

    return schedule


def generate_eoq_strategy(demands, lead_times, initial_stock, avg_demand, unit_cost, shipping_cost):
    """Generate EOQ-based strategy."""
    # Calculate Economic Order Quantity
    annual_demand = avg_demand * 365
    holding_cost_per_unit = unit_cost * 0.2  # 20% of unit cost

    if holding_cost_per_unit > 0:
        eoq = int(((2 * annual_demand * shipping_cost) / holding_cost_per_unit) ** 0.5)  # Round to int
    else:
        eoq = int(avg_demand * 30)  # Default to 30-day supply, rounded to int

    schedule = []
    current_stock = initial_stock
    reorder_threshold = int(avg_demand * 7)  # One week supply

    for day in range(len(demands)):
        # Order when stock is low
        if current_stock <= reorder_threshold:
            schedule.append({
                'day': day,
                'order_quantity': eoq,
                'delivery_day': min(day + int(lead_times[day]), len(demands) - 1),
                'quantity': eoq  # Already int
            })
            current_stock += eoq  # Assume immediate accounting
        else:
            schedule.append({
                'day': day,
                'order_quantity': 0,
                'delivery_day': None,
                'quantity': 0
            })

        current_stock -= demands[day]
        current_stock = max(0, current_stock)

    return schedule


def generate_jit_strategy(demands, lead_times, initial_stock):
    """Generate just-in-time strategy: order exactly what's needed."""
    schedule = []

    for day in range(len(demands)):
        # Order exactly the demand for this day, accounting for lead time
        future_demand = 0
        for future_day in range(day, min(day + int(lead_times[day]) + 1, len(demands))):
            future_demand += demands[future_day]

        # Ensure future_demand is an integer
        future_demand = int(future_demand)

        schedule.append({
            'day': day,
            'order_quantity': future_demand,
            'delivery_day': min(day + int(lead_times[day]), len(demands) - 1),
            'quantity': future_demand if future_demand > 0 else 0  # Already int
        })

    return schedule


def calculate_strategy_cost(strategy, demands, lead_times, initial_stock, unit_cost,
                          margin, shipping_cost, holding_cost_rate, stockout_penalty_multiplier):
    """Calculate total cost for a given strategy."""

    # Simulate the strategy day by day
    current_stock = initial_stock
    total_cost = 0
    inventory_levels = []

    # Track pending orders
    pending_orders = []

    for day in range(len(demands)):
        # Check for arriving orders
        arriving_quantity = 0
        for order in pending_orders[:]:
            if order['delivery_day'] <= day:
                arriving_quantity += order['quantity']
                pending_orders.remove(order)

        current_stock += arriving_quantity

        # Record inventory level for holding cost calculation
        inventory_levels.append(current_stock)

        # Calculate holding cost
        holding_cost = current_stock * unit_cost * holding_cost_rate
        total_cost += holding_cost

        # Handle demand
        if current_stock >= demands[day]:
            current_stock -= demands[day]
        else:
            # Stockout - apply penalty
            stockout_quantity = demands[day] - current_stock
            stockout_penalty = stockout_quantity * margin * stockout_penalty_multiplier
            total_cost += stockout_penalty
            current_stock = 0

        # Place new order if scheduled
        if strategy[day]['quantity'] > 0:
            order_cost = strategy[day]['quantity'] * unit_cost + shipping_cost
            total_cost += order_cost

            # Add to pending orders
            pending_orders.append({
                'quantity': strategy[day]['quantity'],
                'delivery_day': strategy[day]['delivery_day']
            })

    return total_cost


def analyze_strategy_performance(schedule, demands, lead_times, initial_stock, unit_cost,
                               margin, shipping_cost, holding_cost_rate, stockout_penalty_multiplier):
    """Analiza métricas detalladas de rendimiento de una estrategia."""
    # Simulación día a día para obtener métricas
    current_stock = initial_stock
    stockout_days = 0
    inventory_levels = []
    pending_orders = []
    total_cost = 0
    total_holding_cost = 0
    total_order_cost = 0
    total_stockout_cost = 0

    for day in range(len(demands)):
        # Procesar órdenes que llegan
        arriving_quantity = 0
        for order in pending_orders[:]:
            if order['delivery_day'] <= day:
                arriving_quantity += order['quantity']
                pending_orders.remove(order)

        current_stock += arriving_quantity
        inventory_levels.append(current_stock)

        # Calcular costo de almacenamiento
        holding_cost = current_stock * unit_cost * holding_cost_rate
        total_holding_cost += holding_cost
        total_cost += holding_cost

        # Atender demanda
        if current_stock >= demands[day]:
            current_stock -= demands[day]
        else:
            # Stockout
            stockout_days += 1
            stockout_quantity = demands[day] - current_stock
            stockout_cost = stockout_quantity * margin * stockout_penalty_multiplier
            total_stockout_cost += stockout_cost
            total_cost += stockout_cost
            current_stock = 0

        # Colocar nueva orden si está programada
        if day < len(schedule) and schedule[day]['quantity'] > 0:
            order_cost = schedule[day]['quantity'] * unit_cost + shipping_cost
            total_order_cost += order_cost
            total_cost += order_cost

            # Agregar a órdenes pendientes
            if schedule[day]['delivery_day'] is not None:
                pending_orders.append({
                    'quantity': schedule[day]['quantity'],
                    'delivery_day': schedule[day]['delivery_day']
                })

    # Calcular métricas
    avg_inventory = sum(inventory_levels) / len(inventory_levels) if inventory_levels else 0
    service_level = ((len(demands) - stockout_days) / len(demands)) * 100 if demands else 100

    # Calcular costo base (sin optimización)
    baseline_cost = sum(demands) * unit_cost * 1.2  # Asume 20% más de costo sin optimización

    return {
        'total_cost': total_cost,
        'cost_savings': max(0, baseline_cost - total_cost),
        'avg_inventory': avg_inventory,
        'stockout_days': stockout_days,
        'service_level': service_level,
        'strategy_comparison': {
            'current_cost': round(baseline_cost, 2),
            'optimized_cost': round(total_cost, 2),
            'holding_cost': round(total_holding_cost, 2),
            'ordering_cost': round(total_order_cost, 2),
            'stockout_cost': round(total_stockout_cost, 2)
        }
    }


def calculate_current_performance(demands, initial_stock, margin):
    """Calculate performance metrics of current (unoptimized) strategy - simplified version."""
    total_demand = sum(demands)

    # Simple simulation to estimate stockout days
    current_stock = initial_stock
    stockout_days = 0

    for daily_demand in demands:
        if current_stock < daily_demand:
            stockout_days += 1
        current_stock -= daily_demand
        current_stock = max(0, current_stock)

    # Estimate current cost (simplified)
    stockout_penalty = stockout_days * margin * 2  # Rough penalty
    holding_cost = initial_stock * 0.02 * len(demands)  # Rough holding cost

    service_level = ((len(demands) - stockout_days) / len(demands)) * 100 if demands else 100

    return {
        'total_demand': total_demand,
        'stockout_days': stockout_days,
        'total_cost': stockout_penalty + holding_cost,
        'service_level': service_level
    }

@app.route("/results", methods=["GET", "POST"])
def results():
    # Obtener información de productos
    products = execute_query("SELECT SKU, NAME, MARGIN, COST FROM products")
    product_info = {str(prod['SKU']): prod for prod in products}
    all_skus = [str(prod['SKU']) for prod in products]

    # --- RECONSTRUIR best_strategies igual que en /optimize ---
    best_strategies = {}
    for sku in all_skus:
        # Obtener las 10 simulaciones para este SKU
        scenarios = []
        strategies = []
        for iteration in range(1, 11):
            sim_rows = execute_query(
                "SELECT * FROM Simulation WHERE SKU = ? AND Iteration = ? ORDER BY Count",
                sku, iteration
            )
            if not sim_rows:
                continue
            demands = [int(row['DEMAND']) for row in sim_rows]
            lead_times = [int(row['LEAD_TIME']) for row in sim_rows]
            initial_stock = float(sim_rows[0]['STOCK']) + int(sim_rows[0]['DEMAND']) if sim_rows else 0
            unit_cost = float(product_info[sku]['COST']) if product_info[sku]['COST'] else 10
            margin = float(product_info[sku]['MARGIN']) if product_info[sku]['MARGIN'] else 0
            result = optimize_purchase_schedule(
                sku=sku,
                demands=demands,
                lead_times=lead_times,
                initial_stock=initial_stock,
                unit_cost=unit_cost,
                margin=margin,
                shipping_cost=100,
                holding_cost_rate=0.02,
                stockout_penalty_multiplier=2.0,
                simulation_days=len(demands)
            )
            strategies.append(result['optimized_strategy']['purchase_schedule'])
            scenarios.append({
                "demands": demands,
                "lead_times": lead_times,
                "initial_stock": initial_stock
            })

        # Evaluar cada estrategia en todos los escenarios
        avg_costs = []
        for strat_idx, strategy in enumerate(strategies):
            total_cost = 0
            for scenario in scenarios:
                total_cost += calculate_strategy_cost(
                    strategy,
                    scenario['demands'],
                    scenario['lead_times'],
                    scenario['initial_stock'],
                    unit_cost,
                    margin,
                    100,
                    0.02,
                    2.0
                )
            avg_cost = total_cost / len(scenarios) if scenarios else float('inf')
            avg_costs.append(avg_cost)

        # Seleccionar la mejor estrategia
        if avg_costs:
            best_idx = int(np.argmin(avg_costs))
            best_strategies[sku] = {
                "best_iteration": best_idx + 1,
                "average_cost": avg_costs[best_idx],
                "strategy": strategies[best_idx]
            }
        else:
            best_strategies[sku] = None

    # --- AQUÍ SIGUE TU CÓDIGO PARA CALENDAR_MATRIX ---
    sku_list = sorted(best_strategies.keys())
    max_days = max(
        (order["day"] + 1 for info in best_strategies.values() if info for order in info["strategy"]),
        default=30
    )
    calendar_matrix = []
    for day in range(1, max_days + 1):
        row = {"day": day}
        for sku in sku_list:
            row[sku] = ""
            info = best_strategies[sku]
            if info:
                for order in info["strategy"]:
                    if order["day"] + 1 == day and order["quantity"] > 0:
                        row[sku] = order["quantity"]
        calendar_matrix.append(row)

    return render_template(
        "results.html",
        best_strategies=best_strategies,
        product_info=product_info,
        sku_list=sku_list,
        calendar_matrix=calendar_matrix
    )

@app.route("/debug")
def debug_info():
    """Debug route to check environment"""
    try:
        import os
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATABASE)
        return f"""
        <h1>Debug Info</h1>
        <p>Current working directory: {os.getcwd()}</p>
        <p>App file location: {os.path.abspath(__file__)}</p>
        <p>Database path: {db_path}</p>
        <p>Database exists: {os.path.exists(db_path)}</p>
        <p>Directory contents: {os.listdir(os.path.dirname(os.path.abspath(__file__)))}</p>
        """
    except Exception as e:
        return f"Debug error: {str(e)}", 500

if __name__ == "__main__":
    # For development
    app.run(debug=True, host="0.0.0.0", port=5000)