from flask import Flask, flash, redirect, render_template, request, session, jsonify, send_file, make_response, abort
app = Flask(__name__)
# ...existing code...


@app.route("/alerts/low_st.\build-css.batock")
def alerts_low_stock_detail():
    sku = request.args.get("sku", "")
    # You can add validation or extra info loading here if desired
    return render_template("alerts_low_stock.html", sku=sku)

@app.route("/alerts/stock_breaks")
def alerts_stock_breaks_detail():
    sku = request.args.get("sku", "")
    return render_template("alerts_stock_breaks.html", sku=sku)

@app.route("/alerts/stagnated")
def alerts_stagnated_detail():
    sku = request.args.get("sku", "")
    return render_template("alerts_stagnated.html", sku=sku)
import os
import sqlite3
import pandas as pd
import io
import csv
from datetime import datetime, timedelta
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pulp
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpInteger
import functools
import threading
import time
# ...existing code (all other imports remain here, after Flask import)...

app = Flask(__name__)

# Production configuration
import os
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Development configuration for template reloading
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# NexStock - Advanced Inventory Management System
# Version: 2.0.0 - Updated with GitHub integration

DATABASE = "inventory.db"

# Simple in-memory rate limiter (per-IP)
# Note: This is suitable for single-process deployments (dev/staging). For
# production, use a shared store (Redis) or API gateway for distributed rate limiting.
RATE_LIMIT_STORE = {}
RATE_LIMIT_LOCK = threading.Lock()
DEFAULT_RATE_LIMIT = 30  # requests
DEFAULT_RATE_WINDOW = 60  # seconds

def get_client_ip():
    """Return the client IP address from the request.

    Uses X-Forwarded-For if present (first value), otherwise remote_addr.
    """
    try:
        xff = request.headers.get('X-Forwarded-For', '')
        if xff:
            # may be a comma-separated list
            return xff.split(',')[0].strip()
    except Exception:
        pass
    return request.remote_addr or 'unknown'

def rate_limited(limit=DEFAULT_RATE_LIMIT, window=DEFAULT_RATE_WINDOW):
    """Decorator to rate-limit Flask endpoints per client IP.

    Returns 429 JSON when exceeded and sets headers:
    X-RateLimit-Limit, X-RateLimit-Remaining, Retry-After
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ip = get_client_ip()
            now = time.time()
            remaining = limit
            retry_after = 0
            with RATE_LIMIT_LOCK:
                entry = RATE_LIMIT_STORE.get(ip)
                if not entry or (now - entry['start']) >= window:
                    # reset window
                    RATE_LIMIT_STORE[ip] = {'start': now, 'count': 1}
                    remaining = limit - 1
                else:
                    if entry['count'] >= limit:
                        retry_after = int(window - (now - entry['start']))
                        hdrs = {
                            'X-RateLimit-Limit': str(limit),
                            'X-RateLimit-Remaining': '0',
                            'Retry-After': str(retry_after)
                        }
                        return jsonify({'error': 'rate limit exceeded', 'retry_after': retry_after}), 429, hdrs
                    else:
                        entry['count'] += 1
                        remaining = limit - entry['count']

            # Call the real endpoint
            resp = func(*args, **kwargs)

            # Attach headers to response if possible
            try:
                response = make_response(resp)
                response.headers['X-RateLimit-Limit'] = str(limit)
                response.headers['X-RateLimit-Remaining'] = str(remaining)
                return response
            except Exception:
                return resp

        return wrapper
    return decorator

# Add cache-busting timestamp to all templates
@app.context_processor
def inject_cache_buster():
    import time
    return {'cache_buster': int(time.time())}

# Helper function for aggressive cache-busting headers
def add_cache_busting_headers(response):
    """Add comprehensive cache-busting headers to prevent any form of caching"""
    timestamp = int(datetime.now().timestamp())
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0, s-maxage=0, proxy-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    response.headers['ETag'] = f'"{timestamp}"'
    response.headers['Last-Modified'] = datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT')
    response.headers['Vary'] = '*'
    return response

# Serve static files with aggressive cache-busting
@app.route('/static/<path:filename>')
def serve_static(filename):
    from flask import send_from_directory
    response = send_from_directory('static', filename)
    return add_cache_busting_headers(response)

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
        cursor = conn.cursor()
        
        # Create products table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                SKU TEXT PRIMARY KEY,
                NAME TEXT,
                COST REAL,
                MARGIN REAL,
                EXPIRATION INTEGER
            )
        """)
        
        # Create inventory table with auto-increment ID and DATE field
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                SKU TEXT,
                VOLUME REAL,
                DATE TEXT,
                FOREIGN KEY (SKU) REFERENCES products (SKU)
            )
        """)
        
        # Create transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                SKU TEXT,
                VOLUME REAL,
                DOCUMENT_TYPE TEXT,
                DOC_NUMBER TEXT,
                DATE TEXT DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (SKU) REFERENCES products (SKU)
            )
        """)
        
        # Create simulation table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulation (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                SKU TEXT,
                STOCK REAL,
                DEMAND REAL,
                SHIPMENT_MADE REAL,
                Stock_Received REAL,
                LEAD_TIME REAL,
                NOT_BILLED REAL,
                COO REAL,
                Iteration INTEGER,
                FOREIGN KEY (SKU) REFERENCES products (SKU)
            )
        """)
        
        # Insert sample products if none exist
        cursor.execute("SELECT COUNT(*) FROM products")
        if cursor.fetchone()[0] == 0:
            sample_products = [
                ('SKU001', 'Wireless Headphones', 50.00, 0.30, 730),
                ('SKU002', 'Smartphone Case', 15.00, 0.40, 1095),
                ('SKU003', 'Bluetooth Speaker', 80.00, 0.25, 545),
                ('SKU004', 'USB Cable', 10.00, 0.50, 1460),
                ('SKU005', 'Power Bank', 35.00, 0.35, 912)
            ]
            
            cursor.executemany("""
                INSERT INTO products (SKU, NAME, COST, MARGIN, EXPIRATION)
                VALUES (?, ?, ?, ?, ?)
            """, sample_products)
            
            print("Sample products inserted into database.")
        
        conn.commit()
        conn.commit()

        # Ensure additional columns exist for future features (migrations)
        def ensure_column(table, column_def):
            col_name = column_def.split()[0]
            try:
                cursor.execute(f"PRAGMA table_info({table})")
                existing = [row[1] for row in cursor.fetchall()]
                if col_name not in existing:
                    print(f"Adding column {col_name} to {table}", flush=True)
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")
            except Exception as e:
                print(f"Could not ensure column {col_name} on {table}: {e}", flush=True)

        # Products: supplier info and audit timestamps
        ensure_column('products', 'supplier_id INTEGER')
        ensure_column('products', 'supplier_lead_time_days INTEGER DEFAULT 7')
        ensure_column('products', 'created_at TEXT')
        ensure_column('products', 'updated_at TEXT')

        # Inventory: batches, expiry, reservations
        ensure_column('inventory', 'batch_id TEXT')
        ensure_column('inventory', 'expiry_date TEXT')
        ensure_column('inventory', 'location TEXT')
        ensure_column('inventory', 'reserved_qty REAL DEFAULT 0')

        # Transactions: success/failure tracking, status, attempts, source
        ensure_column('transactions', 'success INTEGER DEFAULT 1')
        ensure_column('transactions', 'failure_reason TEXT')
        ensure_column('transactions', 'status TEXT')
        ensure_column('transactions', 'attempt_id TEXT')
        ensure_column('transactions', 'user_id TEXT')

        # Commit any schema changes
        conn.commit()

        # Check final state
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Database initialized successfully. Found {len(tables)} tables: {[table[0] for table in tables]}")
        return True
        
    except Exception as e:
        print(f"Database initialization error: {e}")
        return False
    finally:
        conn.close()

def get_product_info_safe(sku):
    """Safely get product information, handling missing EXPIRATION column"""
    try:
        # Try to get product info with EXPIRATION column
        product_info = execute_query("SELECT SKU, MARGIN, NAME, COST, EXPIRATION FROM products WHERE SKU = ?", sku)
        if product_info:
            return product_info[0]
    except Exception:
        # If EXPIRATION column doesn't exist, try without it
        try:
            product_info = execute_query("SELECT SKU, MARGIN, NAME, COST FROM products WHERE SKU = ?", sku)
            if product_info:
                result = dict(product_info[0])
                result['EXPIRATION'] = 365  # Add default expiration
                return result
        except Exception as e:
            print(f"Error getting product info for SKU {sku}: {e}")
    
    # Return default values if all fails
    return {
        'SKU': sku,
        'MARGIN': 0.0,
        'NAME': 'Unknown',
        'COST': 10.0,
        'EXPIRATION': 365
    }


# --- Helper functions for alert calculations ---
def compute_avg_daily_demand(sku, days=30):
    """Compute average daily demand for an SKU over the last `days` days.

    Demand is derived from transactions that represent outbound/demand documents.
    We treat DOCUMENT_TYPE in ('Invoices', 'Debit Notes', 'Credit Notes') as demand.
    Returns a float >= 0.0.
    """
    try:
        query = (
            "SELECT COALESCE(SUM(ABS(VOLUME)), 0) as sold_sum "
            "FROM transactions "
            "WHERE SKU = ? AND DOCUMENT_TYPE IN ('Invoices', 'Debit Notes', 'Credit Notes') "
            "AND DATE >= date('now', ?)")
        rows = execute_query(query, sku, f"-{int(days)} days")
        sold_sum = float(rows[0]['sold_sum']) if rows and rows[0] and rows[0].get('sold_sum') is not None else 0.0
        avg = sold_sum / float(days) if days > 0 else 0.0
        return float(avg)
    except Exception as e:
        print(f"Error computing avg daily demand for {sku}: {e}")
        return 0.0


def compute_total_stock(sku):
    """Return total available stock (sum of VOLUME) for an SKU from inventory."""
    try:
        rows = execute_query("SELECT COALESCE(SUM(VOLUME), 0) as total_stock FROM inventory WHERE SKU = ?", sku)
        return float(rows[0]['total_stock']) if rows and rows[0] and rows[0].get('total_stock') is not None else 0.0
    except Exception as e:
        print(f"Error computing total stock for {sku}: {e}")
        return 0.0


def get_supplier_lead_time(sku, default_lead=7):
    """Attempt to read supplier/product lead time; fallback to default_lead if not present."""
    try:
        # Try common column names that might store lead time
        rows = execute_query("SELECT COALESCE(LEAD_TIME, SUPPLIER_LEAD_TIME, 0) as lt FROM products WHERE SKU = ?", sku)
        if rows and rows[0] and rows[0].get('lt') not in (None, 0):
            lt = rows[0]['lt']
            try:
                return int(float(lt))
            except Exception:
                return default_lead
        return int(default_lead)
    except Exception:
        return int(default_lead)


def sku_has_min_history(sku, min_days=90):
    """Return True if an SKU has at least `min_days` of sales history (based on earliest sale date).

    We consider demand DOCUMENT_TYPEs the same as in compute_avg_daily_demand.
    """
    try:
        rows = execute_query(
            "SELECT MIN(DATE) as first_sale FROM transactions WHERE SKU = ? AND DOCUMENT_TYPE IN ('Invoices', 'Debit Notes', 'Credit Notes')",
            sku)
        if not rows or rows[0].get('first_sale') is None:
            return False
        first_sale = rows[0]['first_sale']
        try:
            first_date = datetime.fromisoformat(first_sale)
        except Exception:
            # Try parsing YYYY-MM-DD if time not present
            try:
                first_date = datetime.strptime(first_sale.split(' ')[0], '%Y-%m-%d')
            except Exception:
                return False
        delta = datetime.now() - first_date
        return delta.days >= int(min_days)
    except Exception as e:
        print(f"Error checking history for {sku}: {e}")
        return False

# --- end helper functions ---


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
        
        # Initialize database tables
        if init_db():
            print("Database tables initialized", flush=True)
        
        # Get dashboard statistics
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Total Products
        cursor.execute("SELECT COUNT(*) FROM products")
        total_products = cursor.fetchone()[0]
        
        # Total Stock (from inventory table if exists, otherwise use a default)
        try:
            cursor.execute("SELECT SUM(VOLUME) FROM inventory")
            total_stock = cursor.fetchone()[0] or 0
        except:
            total_stock = 0
        
        # Low Stock Items (aggregate by SKU, not batches)
        try:
            # Get total stock per SKU with demand calculations
            low_stock_data = execute_query(
                "SELECT i.SKU, SUM(i.VOLUME) as total_stock, COALESCE(AVG(ABS(t.VOLUME)), 0) as avg_demand FROM inventory i LEFT JOIN transactions t ON i.SKU = t.SKU WHERE i.VOLUME >= 0 GROUP BY i.SKU HAVING total_stock > 0"
            )
            
            low_stock = 0
            for item in low_stock_data:
                total_stock = item['total_stock']
                avg_demand = item['avg_demand']
                
                # Calculate days remaining for this SKU
                days_remaining = total_stock / avg_demand if avg_demand > 0 else float('inf')
                
                # Count as low stock if < 14 days remaining OR < 50 total units
                if days_remaining < 14 or total_stock < 50:
                    low_stock += 1
        except Exception as e:
            print(f"Low stock calculation error: {e}")
            low_stock = 0
        
        # Total Transactions
        try:
            cursor.execute("SELECT COUNT(*) FROM transactions")
            total_transactions = cursor.fetchone()[0] or 0
        except:
            total_transactions = 0
            
        conn.close()
        
        # Get total alerts count (use same logic as /api/alerts/summary)
        try:
            low_stock_alerts = check_low_stock_alerts()
            print(f"[DEBUG] low_stock_alerts: {len(low_stock_alerts)} -> {low_stock_alerts}", flush=True)
            stock_break_alerts = check_stock_breaks()
            print(f"[DEBUG] stock_break_alerts: {len(stock_break_alerts)} -> {stock_break_alerts}", flush=True)
            stagnated_alerts = check_stagnated_items()
            print(f"[DEBUG] stagnated_alerts: {len(stagnated_alerts)} -> {stagnated_alerts}", flush=True)
            total_alerts = len(low_stock_alerts) + len(stock_break_alerts) + len(stagnated_alerts)
            print(f"[DEBUG] total_alerts: {total_alerts}", flush=True)
        except Exception as e:
            print(f"Error getting alerts: {e}")
            total_alerts = 0
        
        # Try to render template with stats
        response = make_response(render_template("index.html", 
                             total_products=total_products,
                             total_stock=total_stock,
                             low_stock=low_stock,
                             total_transactions=total_transactions,
                             total_alerts=total_alerts))
        
        # Add cache-busting headers
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        return response
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
    try:
        products = execute_query("SELECT * FROM products")
        print(f"Portfolio route: Found {len(products)} products")
        if products:
            print(f"First product: {products[0]}")
        # Force template refresh with no-cache headers
        from flask import make_response
        import time
        response = make_response(render_template("portafolio.html", products=products, cache_bust=int(time.time())))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache' 
        response.headers['Expires'] = '0'
        return response
    except Exception as e:
        print(f"Error in portafolio route: {e}")
        return f"Error: {e}", 500

@app.route("/test_portfolio", methods=["GET"])
def test_portfolio():
    products = execute_query("SELECT * FROM products")
    return render_template("test_portfolio.html", products=products)

@app.route("/portfolio_fresh", methods=["GET"])
def portfolio_fresh():
    try:
        products = execute_query("SELECT * FROM products")
        print(f"Fresh Portfolio route: Found {len(products)} products")
        if products:
            print(f"First product data: SKU={products[0].get('SKU')}, COST={products[0].get('COST')}, MARGIN={products[0].get('MARGIN')}")
        # Ultra-aggressive cache busting
        from flask import make_response
        import time
        import time
        timestamp = int(time.time())
        response = make_response(render_template("portfolio_fresh.html", products=products, cache_bust=timestamp))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache' 
        response.headers['Expires'] = '0'
        response.headers['Last-Modified'] = 'Thu, 01 Jan 1970 00:00:00 GMT'
        return response
    except Exception as e:
        print(f"Error in fresh portfolio route: {e}")
        return f"Error: {e}", 500

@app.route("/portfolio_test_new", methods=["GET"])
def portfolio_test_new():
    try:
        products = execute_query("SELECT * FROM products")
        print(f"NEW Test Portfolio: Found {len(products)} products")
        # Maximum cache busting
        from flask import make_response
        import time
        timestamp = int(time.time())
        response = make_response(render_template("portfolio_fresh.html", products=products, cache_bust=timestamp))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0, s-maxage=0, proxy-revalidate'
        response.headers['Pragma'] = 'no-cache' 
        response.headers['Expires'] = 'Thu, 01 Jan 1970 00:00:00 GMT'
        response.headers['Last-Modified'] = 'Thu, 01 Jan 1970 00:00:00 GMT'
        response.headers['ETag'] = f'"cache-bust-{timestamp}"'
        response.headers['Vary'] = '*'
        return response
    except Exception as e:
        print(f"Error in NEW test portfolio route: {e}")
        return f"Error: {e}", 500

@app.route("/debug_portfolio", methods=["GET"])
def debug_portfolio():
    try:
        products = execute_query("SELECT * FROM products")
        print(f"DEBUG Portfolio: Found {len(products)} products")
        for i, product in enumerate(products):
            print(f"Product {i+1}: {dict(product)}")
        return render_template("debug_portfolio.html", products=products)
    except Exception as e:
        print(f"Error in debug portfolio route: {e}")
        return f"Error: {e}", 500

@app.route("/inventory", methods=["GET", "POST"])
def inventory():
    try:
        # Get inventory data with proper date handling
        stock = execute_query(
            "SELECT i.SKU, p.NAME, i.VOLUME, i.DATE as ARRIVAL_DATE, p.EXPIRATION, i.ID as BATCH_ID FROM inventory i JOIN products p ON i.SKU = p.SKU WHERE i.VOLUME > 0 ORDER BY i.SKU, i.DATE ASC"
        )
        
        # Calculate days to expire in Python (safe approach)
        from datetime import datetime
        
        for item in stock:
            # Get expiration period from products table (total shelf life)
            expiration_days = item.get('EXPIRATION', None)
            arrival_date_str = item.get('ARRIVAL_DATE', None)
            
            if expiration_days and arrival_date_str:
                try:
                    # Parse arrival date
                    arrival_date = datetime.strptime(arrival_date_str, '%Y-%m-%d')
                    today = datetime.now()
                    
                    # Calculate aging time (how long this batch has been in inventory)
                    aging_days = (today - arrival_date).days
                    
                    # Calculate remaining shelf life
                    remaining_days = expiration_days - aging_days
                    
                    item['DAYS_TO_EXPIRE'] = remaining_days
                        
                except Exception:
                    item['DAYS_TO_EXPIRE'] = None
            else:
                item['DAYS_TO_EXPIRE'] = None
        
        # Get summary stats by SKU
        summary = execute_query(
            "SELECT p.SKU, p.NAME, COALESCE(SUM(i.VOLUME), 0) as TOTAL_VOLUME, COUNT(i.ID) as BATCH_COUNT FROM products p LEFT JOIN inventory i ON p.SKU = i.SKU AND i.VOLUME > 0 GROUP BY p.SKU, p.NAME ORDER BY p.SKU"
        )
        
        response = make_response(render_template("inventory.html", stock=stock, summary=summary))
        
        # Add cache-busting headers
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        return response
        
    except Exception as e:
        # Return a simple error page instead of crashing
        return f"<h1>Inventory Temporarily Unavailable</h1><p>Error: {str(e)}</p><p><a href='/'>Return to Dashboard</a></p>", 500


@app.route("/debug-inventory")
def debug_inventory():
    return "Debug inventory route works! Version 2.0"

@app.route("/debug-stats")
def debug_stats():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the same stats as dashboard
        cursor.execute("SELECT COUNT(*) FROM products")
        total_products = cursor.fetchone()[0]
        
        try:
            cursor.execute("SELECT SUM(VOLUME) FROM inventory")
            total_stock = cursor.fetchone()[0] or 0
        except:
            total_stock = 0
        
        try:
            cursor.execute("SELECT COUNT(DISTINCT SKU) FROM inventory WHERE VOLUME < 10")
            low_stock = cursor.fetchone()[0] or 0
        except:
            low_stock = 0
        
        try:
            cursor.execute("SELECT COUNT(*) FROM transactions")
            total_transactions = cursor.fetchone()[0] or 0
        except:
            total_transactions = 0
            
        conn.close()
        
        return f"""
        <h1>Debug Stats</h1>
        <p>Total Products: {total_products}</p>
        <p>Total Stock: {total_stock}</p>
        <p>Low Stock Items: {low_stock}</p>
        <p>Total Transactions: {total_transactions}</p>
        <hr>
        <p><a href="/">Dashboard</a> | <a href="/inventory">Inventory</a></p>
        """
        
    except Exception as e:
        return f"Error: {e}"

@app.route("/test-xyz")
def test_xyz():
    print("TEST XYZ ROUTE CALLED!")
    return "Test XYZ works!"


@app.route("/simulate", methods=["GET"])
def simulation():
    """Renders the simulation page and loads existing simulation data if available."""
    
    # Check if there's existing simulation data
    existing_data = None
    try:
        # Check if simulation table has data
        sim_count = execute_query("SELECT COUNT(*) as count FROM simulation")
        if sim_count and sim_count[0]['count'] > 0:
            # Get the most recent simulation parameters and results
            # We'll reconstruct a basic results structure from the simulation table
            sim_data = execute_query("""
                SELECT DISTINCT SKU 
                FROM simulation 
                ORDER BY SKU
            """)
            
            if sim_data:
                # Create a simplified results structure for display
                results = []
                for sku_row in sim_data:
                    sku = sku_row['SKU']
                    
                    # Get basic stats for this SKU from simulation data
                    sku_stats = execute_query("""
                        SELECT 
                            AVG(DEMAND) as avg_demand,
                            MIN(DEMAND) as min_demand,
                            MAX(DEMAND) as max_demand,
                            AVG(STOCK) as avg_stock,
                            MIN(STOCK) as min_stock,
                            MAX(STOCK) as max_stock,
                            AVG(LEAD_TIME) as avg_lead_time,
                            SUM(NOT_BILLED) as total_not_billed
                        FROM simulation 
                        WHERE SKU = ?
                    """, sku)
                    
                    if sku_stats:
                        stats = sku_stats[0]
                        # Get product name
                        product_info = execute_query("SELECT NAME FROM products WHERE SKU = ?", sku)
                        product_name = product_info[0]['NAME'] if product_info else f"Product {sku}"
                        
                        results.append({
                            "sku": sku,
                            "product_name": product_name,
                            "mean_demand": round(float(stats['avg_demand'] or 0), 1),
                            "min_demand": int(stats['min_demand'] or 0),
                            "max_demand": int(stats['max_demand'] or 0),
                            "mean_stock": round(float(stats['avg_stock'] or 0), 1),
                            "min_stock": round(float(stats['min_stock'] or 0), 1),
                            "max_stock": round(float(stats['max_stock'] or 0), 1),
                            "avg_lead_time": round(float(stats['avg_lead_time'] or 0), 1),
                            "total_unfulfilled": round(float(stats['total_not_billed'] or 0), 1),
                            "days_negative_avg": 0  # This would require more complex calculation
                        })
                
                if results:
                    existing_data = {
                        "status": "success",
                        "message": f"Loaded existing simulation data for {len(results)} SKUs",
                        "results": results,
                        "simulation_size": 30,  # Default values since we don't store these
                        "lead_time_distribution": "normal",
                        "num_iterations": 1
                    }
                    
    except Exception as e:
        print(f"Error loading existing simulation data: {e}")
        existing_data = None
    
    response = make_response(render_template("simulate.html", existing_data=existing_data))
    # Add cache-busting headers
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route("/run_simulation", methods=["POST"])
def run_simulation():
    """Runs the simulation and returns JSON results."""

    try:
        # Clear previous simulation data
        print("Clearing previous simulation data...", flush=True)
        execute_query("DELETE FROM simulation")
        print("Previous simulation data cleared.", flush=True)

        # Get and validate input data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        size = data.get('size', 30)
        distribution = data.get('lead_time_distribution', 'normal')
        param1 = data.get('lead_time_param1', 7.0)
        param2 = data.get('lead_time_param2', 2.0)
        num_iterations = data.get('num_iterations', 10)
        waste_cost_rate = data.get('waste_cost_rate', 80.0)

        # Validate and convert inputs
        try:
            size = int(size)
            param1 = float(param1)
            param2 = float(param2)
            num_iterations = int(num_iterations)
            waste_cost_rate = float(waste_cost_rate)

            if size <= 0 or size > 180:  # Reduced limit to avoid OOM
                return jsonify({"error": "Size must be between 1 and 180 days"}), 400

            if distribution not in ['normal', 'uniform']:
                return jsonify({"error": "Distribution must be 'normal' or 'uniform'"}), 400

            if num_iterations < 1 or num_iterations > 10:  # Reduced limit to avoid OOM
                return jsonify({"error": "Number of iterations must be between 1 and 10"}), 400

            if waste_cost_rate < 0 or waste_cost_rate > 100:
                return jsonify({"error": "Waste cost rate must be between 0% and 100%"}), 400

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
            all_simulation_inserts = []

            for iteration in range(1, num_iterations + 1):
                simulation_inserts = []
                

                for sku_row in sim_parameters:
                    sku = sku_row["SKU"]
                    print(f"Processing SKU: {sku}", flush=True)

                    if distribution == 'normal':
                        lead_times = rng.normal(param1, param2, size)
                        lead_times = np.maximum(lead_times, 0.1)  # Ensure positive
                    else:  # uniform
                        lead_times = rng.uniform(param1, param2, size)

                    lead_times = np.round(lead_times).astype(int).tolist()    

                    # Get product information including margin, cost, and expiration
                    product_info = get_product_info_safe(sku)
                    
                    margin = float(product_info.get("MARGIN", 0.0))
                    unit_cost = float(product_info.get("COST", 10.0))
                    product_name = product_info.get("NAME", "Unknown")
                    expiration_days = int(product_info.get("EXPIRATION", 365))

                    print(f"SKU {sku} ({product_name}): Margin = {margin}, Cost = {unit_cost}, Expires in {expiration_days} days", flush=True)

                    # Get initial inventory for this SKU
                    initial_inventory = execute_query(
                        "SELECT SKU, SUM(VOLUME) as STOCK FROM inventory WHERE SKU = ?", sku)

                    if initial_inventory and initial_inventory[0]["STOCK"] is not None:
                        initial_stock = float(initial_inventory[0]["STOCK"])
                    else:
                        initial_stock = 0.0

                    print(f"SKU {sku}: Initial stock = {initial_stock}", flush=True)

                    # Get demand-side transactions only (exclude supply-side transactions)
                    # Consider Invoices, Debit Notes, and Credit Notes for demand forecasting
                    sku_data = execute_query(
                        "SELECT VOLUME FROM transactions WHERE SKU = ? AND DOCUMENT_TYPE IN ('Invoices', 'Debit Notes', 'Credit Notes')", 
                        sku
                    )
                    # Convert to absolute values since we're analyzing demand magnitude
                    volumes = [abs(float(row["VOLUME"])) for row in sku_data]
                    print(f"SKU {sku}: Found {len(volumes)} demand transactions: {volumes}", flush=True)

                    if len(volumes) < 1:
                        print(f"SKU {sku}: Skipping due to no demand data", flush=True)
                        continue  # Skip SKUs with no demand data

                    # Calculate demand statistics
                    avg_vol = sum(volumes) / len(volumes)
                    
                    if len(volumes) > 1:
                        variance = sum((v - avg_vol) ** 2 for v in volumes) / (len(volumes) - 1)
                        sd_vol = variance ** 0.5 if variance > 0 else avg_vol * 0.1
                    else:
                        # For single transaction, use 10% of average as standard deviation
                        sd_vol = avg_vol * 0.1

                    # Generate demand values
                    demand_array = rng.normal(loc=avg_vol, scale=sd_vol, size=size)
                    demand_array = np.maximum(demand_array, 0)
                    demand_integers = np.round(demand_array).astype(int).tolist()

                    # Initialize inventory tracking with age
                    # Each inventory item has [quantity, age_in_days]
                    inventory_batches = [[initial_stock, 0]] if initial_stock > 0 else []
                    stock_levels = []
                    total_waste_cost = 0.0
                    total_waste_quantity = 0.0

                    for day_idx, daily_demand in enumerate(demand_integers):
                        # Age all inventory batches by 1 day
                        for batch in inventory_batches:
                            batch[1] += 1
                        
                        # Remove expired inventory and calculate waste cost
                        expired_batches = []
                        remaining_batches = []
                        
                        for batch in inventory_batches:
                            if batch[1] >= expiration_days:  # Product has expired
                                expired_batches.append(batch)
                                waste_quantity = batch[0]
                                waste_cost = waste_quantity * unit_cost * (waste_cost_rate / 100.0)
                                total_waste_cost += waste_cost
                                total_waste_quantity += waste_quantity
                            else:
                                remaining_batches.append(batch)
                        
                        inventory_batches = remaining_batches
                        
                        # Calculate current stock level after removing expired items
                        current_stock = sum(batch[0] for batch in inventory_batches)
                        stock_levels.append(round(current_stock, 2))
                        
                        # Satisfy demand using FIFO (oldest first)
                        remaining_demand = daily_demand
                        new_batches = []
                        
                        for batch in inventory_batches:
                            if remaining_demand <= 0:
                                new_batches.append(batch)
                            elif batch[0] <= remaining_demand:
                                remaining_demand -= batch[0]
                                # This batch is completely consumed
                            else:
                                # Partial consumption of this batch
                                batch[0] -= remaining_demand
                                remaining_demand = 0
                                new_batches.append(batch)
                        
                        inventory_batches = new_batches
                        current_stock = sum(batch[0] for batch in inventory_batches)

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
                        "waste_cost": round(float(total_waste_cost), 2),
                        "waste_quantity": round(float(total_waste_quantity), 2),
                        "expiration_days": expiration_days,
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
                            0,  # SHIPMENT_MADE (ajusta si tienes el dato)
                            0,  # Stock_Received (ajusta si tienes el dato)
                            lead_time_value,
                            not_billed,
                            0,  # COO (ajusta si tienes el dato)
                            iteration
                        ))
                
                # Add this iteration's data to the complete list
                all_simulation_inserts.extend(simulation_inserts)
            
            # Perform batch insert using transaction
            if all_simulation_inserts:
                operations = []
                for insert_data in all_simulation_inserts:
                    operations.append(("""
                        INSERT INTO simulation
                        (SKU, STOCK, DEMAND, SHIPMENT_MADE, Stock_Received, LEAD_TIME, NOT_BILLED, COO, Iteration)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, insert_data))
                execute_transaction(operations)

            print(f"Simulation completed. Inserted {len(all_simulation_inserts)} records.", flush=True)

            # --- NEW BLOCK TO ADD STATISTICS FOR THE 10 ITERATIONS ---
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
                "days_negative": [],
                "waste_cost": [],
                "waste_quantity": []
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
                sku_stats[sku]["waste_cost"].append(r["waste_cost"])
                sku_stats[sku]["waste_quantity"].append(r["waste_quantity"])

            # Calculate aggregated statistics by SKU
            aggregated_results = []
            for sku, data in sku_stats.items():
                # Unify all generated values and stock_levels from the 10 simulations for global statistics
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
                    "waste_cost_rate": waste_cost_rate,
                    "expiration_days": data["expiration_days"] if "expiration_days" in data else 365,
                    "total_waste_cost": round(float(np.sum(data["waste_cost"])), 2) if data["waste_cost"] else 0,
                    "total_waste_quantity": round(float(np.sum(data["waste_quantity"])), 2) if data["waste_quantity"] else 0,
                    # Global statistics over the 10 runs
                    "min_demand": int(np.min(all_demands)) if all_demands else 0,
                    "avg_demand": float(np.mean(all_demands)) if all_demands else 0,
                    "max_demand": int(np.max(all_demands)) if all_demands else 0,
                    "min_stock": float(np.min(all_stocks)) if all_stocks else 0,
                    "max_stock": float(np.max(all_stocks)) if all_stocks else 0,
                    "days_negative_avg": float(np.mean(data["days_negative"])) if data["days_negative"] else 0,
                    # Statistics per simulation (averages of the 10 runs)
                    "mean_demand_per_run": float(np.mean(data["mean_demand"])) if data["mean_demand"] else 0,
                    "min_demand_per_run": float(np.mean(data["min_demand"])) if data["min_demand"] else 0,
                    "max_demand_per_run": float(np.mean(data["max_demand"])) if data["max_demand"] else 0,
                    "min_stock_per_run": float(np.mean(data["min_stock"])) if data["min_stock"] else 0,
                    "max_stock_per_run": float(np.mean(data["max_stock"])) if data["max_stock"] else 0,
                })

            return jsonify({
                "status": "success",
                "message": f"Generated simulation data for {len(aggregated_results)} SKUs (aggregated over {num_iterations} iterations)",
                "total_records": len(aggregated_results),
                "simulation_size": size,
                "lead_time_distribution": distribution,
                "num_iterations": num_iterations,
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

def process_transaction(sku, volume, doc_type, doc_number=None):
    # Process a transaction and update inventory accordingly
    try:
        # (Lógica original de la función, todo el cuerpo actual va indentado dentro de este try)
        # Ensure positive volume for inbound transactions
        volume = abs(volume) if doc_type in ["Receipts"] else volume
        if doc_type in ["Invoices", "Credit Notes", "Debit Notes"]:
            volume = -abs(volume)
        elif doc_type in ["Purchase Orders (PO)"]:
            volume = abs(volume)
            affects_inventory = False
        else:
            affects_inventory = True

        # Generate document number if not provided
        if doc_number is None:
            last_doc = execute_query("SELECT MAX(DOC_NUMBER) as max_doc FROM transactions WHERE DOC_NUMBER GLOB '[0-9]*'")
            doc_number = (int(last_doc[0]['max_doc']) if last_doc and last_doc[0]['max_doc'] else 0) + 1

        # Insert transaction record
        current_date = datetime.now().date().isoformat()
        transaction_id = execute_query(
            "INSERT INTO transactions (SKU, VOLUME, DOCUMENT_TYPE, DOC_NUMBER, DATE) VALUES (?, ?, ?, ?, ?)",
            sku, volume, doc_type, doc_number, current_date
        )

        # Get the inserted transaction ID
        if transaction_id:
            last_row = execute_query("SELECT last_insert_rowid() as id")
            transaction_id = last_row[0]['id'] if last_row else None

        # Update inventory based on transaction type - but only if it affects physical inventory
        if 'affects_inventory' not in locals():
            affects_inventory = True
        if affects_inventory:
            if volume > 0:
                execute_query(
                    "INSERT INTO inventory (SKU, DATE, VOLUME) VALUES (?, ?, ?)",
                    sku, current_date, volume
                )
            else:
                remaining_to_remove = abs(volume)
                inventory_batches = execute_query(
                    "SELECT ID, VOLUME FROM inventory WHERE SKU = ? AND VOLUME > 0 ORDER BY DATE ASC",
                    sku
                )
                for batch in inventory_batches:
                    if remaining_to_remove <= 0:
                        break
                    batch_id = batch['ID']
                    batch_volume = batch['VOLUME']
                    if batch_volume <= remaining_to_remove:
                        execute_query("DELETE FROM inventory WHERE ID = ?", batch_id)
                        remaining_to_remove -= batch_volume
                    else:
                        new_batch_volume = batch_volume - remaining_to_remove
                        execute_query("UPDATE inventory SET VOLUME = ? WHERE ID = ?", new_batch_volume, batch_id)
                        remaining_to_remove = 0
                if remaining_to_remove > 0:
                    return {
                        "error": f"Insufficient inventory. Could not remove {remaining_to_remove} additional units.",
                        "success": False,
                        "partial_completion": True,
                        "removed": abs(volume) - remaining_to_remove
                    }

        total_stock = execute_query(
            "SELECT SUM(VOLUME) as total_stock FROM inventory WHERE SKU = ?",
            sku
        )
        new_stock_level = total_stock[0]['total_stock'] if total_stock and total_stock[0]['total_stock'] else 0

        if affects_inventory:
            message = f"Transaction recorded: {abs(volume)} units {'added to' if volume > 0 else 'removed from'} inventory"
        else:
            message = f"Transaction recorded: {doc_type} for {abs(volume)} units (no inventory impact)"

        return {
            "success": True,
            "sku": sku,
            "transaction_volume": volume,
            "document_type": doc_type,
            "document_number": doc_number,
            "new_stock_level": new_stock_level,
            "date": current_date,
            "affects_inventory": affects_inventory,
            "message": message
        }
    except Exception as e:
        return {"error": str(e), "success": False}
        # - Purchase Orders (PO): Just a commitment, NO inventory impact (record only)
        
        # Determine transaction sign and inventory impact based on document type
        # Only certain document types should actually affect physical inventory:
        # - Receipts: Goods received, increase inventory (positive)
        # - Invoices: Goods sold/shipped, decrease inventory (negative) 
        # - Credit Notes: Returns from customers, decrease inventory (negative)
        # - Debit Notes: Adjustments, typically decrease inventory (negative)
        # - Purchase Orders (PO): Just a commitment, NO inventory impact (record only)
        
        inventory_affecting_inbound = ["Receipts"]  # Only receipts increase inventory
        inventory_affecting_outbound = ["Invoices", "Credit Notes", "Debit Notes"]  # These decrease inventory
        non_inventory_affecting = ["Purchase Orders (PO)"]  # These are just records, no inventory impact
        
        affects_inventory = True
        
        if doc_type in inventory_affecting_inbound:
            # Ensure positive volume for inbound transactions
            volume = abs(volume)
        elif doc_type in inventory_affecting_outbound:
            # Ensure negative volume for outbound transactions
            volume = -abs(volume)
        elif doc_type in non_inventory_affecting:
            # Purchase orders don't affect inventory, just record the transaction
            volume = abs(volume)  # Keep positive for record purposes, but won't update inventory
            affects_inventory = False
        else:
            # For unknown document types, keep the sign as provided
            pass
        
        # Generate document number if not provided
        if doc_number is None:
            # Get next document number
            last_doc = execute_query("SELECT MAX(DOC_NUMBER) as max_doc FROM transactions WHERE DOC_NUMBER GLOB '[0-9]*'")
            doc_number = (int(last_doc[0]['max_doc']) if last_doc and last_doc[0]['max_doc'] else 0) + 1
        
        # Insert transaction record
        current_date = datetime.now().date().isoformat()
        transaction_id = execute_query(
            sku, volume, doc_type, doc_number, current_date
        )
        
        # Get the inserted transaction ID
        if transaction_id:
            last_row = execute_query("SELECT last_insert_rowid() as id")
            transaction_id = last_row[0]['id'] if last_row else None
        
        # Update inventory based on transaction type - but only if it affects physical inventory
        if affects_inventory:
            if volume > 0:
                # Inbound transaction - add new inventory batch
                execute_query(
                    "INSERT INTO inventory (SKU, DATE, VOLUME) VALUES (?, ?, ?)",
                    sku, current_date, volume
                )
            else:
                # Outbound transaction - reduce from existing inventory (FIFO)
                remaining_to_remove = abs(volume)
                
                # Get inventory batches sorted by date (FIFO - First In, First Out)
                inventory_batches = execute_query(
                    "SELECT ID, VOLUME FROM inventory WHERE SKU = ? AND VOLUME > 0 ORDER BY DATE ASC",
                    sku
                )
                
                for batch in inventory_batches:
                    if remaining_to_remove <= 0:
                        break
                    
                    batch_id = batch['ID']
                    batch_volume = batch['VOLUME']
                    
                    if batch_volume <= remaining_to_remove:
                        # Remove entire batch
                        execute_query("DELETE FROM inventory WHERE ID = ?", batch_id)
                        remaining_to_remove -= batch_volume
                    else:
                        # Partially reduce batch
                        new_batch_volume = batch_volume - remaining_to_remove
                        execute_query("UPDATE inventory SET VOLUME = ? WHERE ID = ?", new_batch_volume, batch_id)
                        remaining_to_remove = 0
                
                # Check if we couldn't fulfill the entire outbound transaction
                if remaining_to_remove > 0:
                    return {
                        "error": f"Insufficient inventory. Could not remove {remaining_to_remove} additional units.",
                        "success": False,
                        "partial_completion": True,
                        "removed": abs(volume) - remaining_to_remove
                    }
        
        # Calculate new total stock level
        total_stock = execute_query(
            "SELECT SUM(VOLUME) as total_stock FROM inventory WHERE SKU = ?",
            sku
        )
        new_stock_level = total_stock[0]['total_stock'] if total_stock and total_stock[0]['total_stock'] else 0
        
        # Create appropriate message based on whether inventory was affected
        if affects_inventory:
            message = f"Transaction recorded: {abs(volume)} units {'added to' if volume > 0 else 'removed from'} inventory"
        else:
            message = f"Transaction recorded: {doc_type} for {abs(volume)} units (no inventory impact)"
        
        return {
            "success": True,
            "sku": sku,
            "transaction_volume": volume,
            "document_type": doc_type,
            "document_number": doc_number,
            "new_stock_level": new_stock_level,
            "date": current_date,
            "affects_inventory": affects_inventory,
            "message": message
        }
        
    except Exception as e:
        return {"error": str(e), "success": False}

@app.route("/api/transactions", methods=["POST"])
def create_transaction():
    """Create a new transaction and update inventory"""
    try:
        data = request.get_json()
        sku = data.get('sku')
        volume = float(data.get('volume', 0))
        doc_type = data.get('document_type', 'Invoices')
        doc_number = data.get('document_number')
        
        if not sku or volume == 0:
            return jsonify({"error": "SKU and volume are required"}), 400
        
        # Validate document type
        valid_doc_types = ['Invoices', 'Receipts', 'Purchase Orders (PO)', 'Credit Notes', 'Debit Notes']
        if doc_type not in valid_doc_types:
            return jsonify({"error": f"Invalid document type. Must be one of: {valid_doc_types}"}), 400
        
        # Process the transaction
        result = process_transaction(sku, volume, doc_type, doc_number)
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/transactions", methods=["GET", "POST"])
def transactions():
    if request.method == "POST":
        # Handle form submission for new transaction
        try:
            sku = request.form.get('sku')
            volume = float(request.form.get('volume', 0))
            doc_type = request.form.get('document_type', 'Invoices')
            doc_number = request.form.get('document_number')
            
            if not sku or volume == 0:
                flash("SKU and volume are required", "error")
            else:
                result = process_transaction(sku, volume, doc_type, doc_number)
                if result.get('success'):
                    flash(result['message'], "success")
                else:
                    flash(result.get('error', 'Transaction failed'), "error")
                    
        except Exception as e:
            flash(f"Error processing transaction: {str(e)}", "error")
    
    # Get all transactions for display
    trades = execute_query("SELECT * FROM transactions ORDER BY DATE DESC, DOC_NUMBER DESC")
    return render_template("transactions.html", trades=trades)

@app.route("/optimize", methods=["GET", "POST"])
def optimization():
    print(f"DEBUG: /optimize route called, method={request.method}", flush=True)
    # For GET requests, simply show the page with the button
    if request.method == "GET":
        # Check if simulation data exists
        try:
            sim_count = execute_query("SELECT COUNT(*) as count FROM simulation")
            print(f"GET /optimize: Found {sim_count[0]['count']} simulation records", flush=True)
            if sim_count[0]['count'] == 0:
                return render_template("optimize.html", error="No simulation data found. Please run a simulation first.", all_optimizations={})
            else:
                # Show some info about available data
                skus_with_data = execute_query("SELECT DISTINCT SKU FROM simulation")
                iterations = execute_query("SELECT DISTINCT Iteration FROM simulation ORDER BY Iteration DESC LIMIT 3")
                print(f"GET /optimize: Available SKUs: {[s['SKU'] for s in skus_with_data]}, Recent iterations: {[i['Iteration'] for i in iterations]}", flush=True)
        except Exception as e:
            print(f"GET /optimize: Database error: {e}", flush=True)
            return render_template("optimize.html", error=f"Database error: {str(e)}", all_optimizations={})
        
        response = make_response(render_template("optimize.html", all_optimizations={}))
        # Add cache-busting headers
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    
    # Only execute optimization on POST requests (when button is clicked)
    try:
        print("DEBUG: /optimize POST reached (very top of handler)", flush=True)
        print("POST /optimize: Starting optimization process...", flush=True)
        # Get parameters from form or use defaults
        shipping_cost = float(request.form.get('shipping_cost', 100))  # Fixed cost per shipment
        holding_cost_rate = float(request.form.get('holding_cost_rate', 1)) / 100  # Convert percentage to decimal
        stockout_penalty_multiplier = float(request.form.get('stockout_penalty', 2.0))  # Multiplier over margin to penalize stockouts
        
        print(f"POST /optimize: Parameters - shipping_cost={shipping_cost}, holding_cost_rate={holding_cost_rate}, stockout_penalty={stockout_penalty_multiplier}", flush=True)

        # Get product info safely
        products = execute_query("SELECT SKU, NAME, MARGIN, COST FROM products")
        product_info = {}
        for prod in products:
            # Use the safe function to get complete product info including EXPIRATION
            product_info[prod['SKU']] = get_product_info_safe(prod['SKU'])

        # Get all SKUs
        all_skus = [prod['SKU'] for prod in products]

        # Prepare results structure
        all_optimizations = {}

        # Get the maximum iteration number from the database
        max_iteration_result = execute_query("SELECT MAX(Iteration) as max_iter FROM simulation")
        max_iterations = max_iteration_result[0]['max_iter'] if max_iteration_result and max_iteration_result[0]['max_iter'] else 10

        for sku in all_skus:
            all_optimizations[sku] = []
            for iteration in range(1, max_iterations + 1):
                # Get simulation data for this SKU and iteration
                sim_rows = execute_query(
                    "SELECT * FROM simulation WHERE SKU = ? AND Iteration = ?",
                    sku, iteration
                )
                if not sim_rows:
                    print(f"No simulation data found for SKU {sku}, iteration {iteration}", flush=True)
                    continue

                demands = [int(row['DEMAND']) for row in sim_rows]
                lead_times = [int(row['LEAD_TIME']) for row in sim_rows]
                initial_stock = float(sim_rows[0]['STOCK']) + int(sim_rows[0]['DEMAND']) if sim_rows else 0

                unit_cost = float(product_info[sku].get('COST', 10))
                margin = float(product_info[sku].get('MARGIN', 0))
                expiration_days = int(product_info[sku].get('EXPIRATION', 365))

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
                    simulation_days=len(demands),
                    expiration_days=expiration_days,
                    waste_cost_rate=80.0  # Default waste cost rate for optimization
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

        # Calculate metrics for summary using AVERAGE per iteration, not total sum
        # Since you would only implement ONE strategy, not all iterations
        total_service_level = 0
        total_iterations = 0
        total_savings = 0
        
        # Initialize cost accumulators
        total_costs = {
            'ordering_cost': 0,
            'purchase_cost': 0,
            'holding_cost': 0,
            'stockout_cost': 0,
            'waste_cost': 0,
            'total_cost': 0
        }

        # Collect costs per iteration to then average
        iteration_costs = {}
        
        for runs in all_optimizations.values():
            for run in runs:
                iteration = run['iteration']
                if iteration not in iteration_costs:
                    iteration_costs[iteration] = {
                        'ordering_cost': 0,
                        'purchase_cost': 0,
                        'holding_cost': 0,
                        'stockout_cost': 0,
                        'waste_cost': 0,
                        'total_cost': 0,
                        'service_level': 0,
                        'sku_count': 0
                    }
                
                total_service_level += run['result']['optimized_strategy']['service_level']
                total_iterations += 1
                
                # Accumulate costs per iteration (sum of all SKUs in this iteration)
                breakdown = run['result']['optimized_strategy']['cost_breakdown']
                iteration_costs[iteration]['ordering_cost'] += breakdown.get('ordering_cost', 0)
                iteration_costs[iteration]['purchase_cost'] += breakdown.get('purchase_cost', 0)
                iteration_costs[iteration]['holding_cost'] += breakdown.get('holding_cost', 0)
                iteration_costs[iteration]['stockout_cost'] += breakdown.get('stockout_cost', 0)
                iteration_costs[iteration]['waste_cost'] += breakdown.get('waste_cost', 0)
                iteration_costs[iteration]['total_cost'] += run['result']['optimized_strategy']['total_cost']
                iteration_costs[iteration]['service_level'] += run['result']['optimized_strategy']['service_level']
                iteration_costs[iteration]['sku_count'] += 1
                
        # Calculate average costs between iterations (cost of implementing ONE complete strategy)
        num_iterations = len(iteration_costs)
        if num_iterations > 0:
            for cost_type in total_costs.keys():
                total_costs[cost_type] = sum(iter_cost[cost_type] for iter_cost in iteration_costs.values()) / num_iterations
            
            # Calculate savings based on average purchase cost
            total_savings = total_costs['purchase_cost'] * 0.1

        # Calculate percentages of each cost type
        cost_percentages = {}
        if total_costs['total_cost'] > 0:
            for cost_type, amount in total_costs.items():
                if cost_type != 'total_cost':
                    cost_percentages[cost_type] = (amount / total_costs['total_cost']) * 100

        # Calculate per-SKU cost breakdowns for filtering
        sku_cost_breakdowns = {}
        for sku, runs in all_optimizations.items():
            if runs:
                sku_iteration_costs = {}
                for run in runs:
                    iteration = run['iteration']
                    breakdown = run['result']['optimized_strategy']['cost_breakdown']
                    
                    if iteration not in sku_iteration_costs:
                        sku_iteration_costs[iteration] = {
                            'ordering_cost': 0,
                            'purchase_cost': 0,
                            'holding_cost': 0,
                            'stockout_cost': 0,
                            'waste_cost': 0,
                            'total_cost': 0
                        }
                    
                    # Add costs for this iteration
                    sku_iteration_costs[iteration]['ordering_cost'] += breakdown.get('ordering_cost', 0)
                    sku_iteration_costs[iteration]['purchase_cost'] += breakdown.get('purchase_cost', 0)
                    sku_iteration_costs[iteration]['holding_cost'] += breakdown.get('holding_cost', 0)
                    sku_iteration_costs[iteration]['stockout_cost'] += breakdown.get('stockout_cost', 0)
                    sku_iteration_costs[iteration]['waste_cost'] += breakdown.get('waste_cost', 0)
                    sku_iteration_costs[iteration]['total_cost'] += run['result']['optimized_strategy']['total_cost']
                
                # Average costs across iterations for this SKU
                num_sku_iterations = len(sku_iteration_costs)
                if num_sku_iterations > 0:
                    sku_avg_costs = {
                        'ordering_cost': sum(iter_cost['ordering_cost'] for iter_cost in sku_iteration_costs.values()) / num_sku_iterations,
                        'purchase_cost': sum(iter_cost['purchase_cost'] for iter_cost in sku_iteration_costs.values()) / num_sku_iterations,
                        'holding_cost': sum(iter_cost['holding_cost'] for iter_cost in sku_iteration_costs.values()) / num_sku_iterations,
                        'stockout_cost': sum(iter_cost['stockout_cost'] for iter_cost in sku_iteration_costs.values()) / num_sku_iterations,
                        'waste_cost': sum(iter_cost['waste_cost'] for iter_cost in sku_iteration_costs.values()) / num_sku_iterations,
                        'total_cost': sum(iter_cost['total_cost'] for iter_cost in sku_iteration_costs.values()) / num_sku_iterations
                    }
                    
                    # Calculate percentages for this SKU
                    sku_percentages = {}
                    if sku_avg_costs['total_cost'] > 0:
                        for cost_type, amount in sku_avg_costs.items():
                            if cost_type != 'total_cost':
                                sku_percentages[cost_type] = (amount / sku_avg_costs['total_cost']) * 100
                    
                    sku_cost_breakdowns[sku] = {
                        'cost_breakdown': sku_avg_costs,
                        'cost_percentages': sku_percentages,
                        'iterations': num_sku_iterations
                    }

        # Get all available SKUs from products table for the filter dropdown
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT SKU FROM products ORDER BY CAST(SKU AS INTEGER)")
            all_available_skus = [row[0] for row in cursor.fetchall()]
            conn.close()
        except Exception as e:
            print(f"Error fetching SKUs from products table: {e}")
            all_available_skus = list(sku_cost_breakdowns.keys())  # Fallback to optimization results

        # For SKUs without optimization data, create empty cost breakdowns
        for sku in all_available_skus:
            if sku not in sku_cost_breakdowns:
                sku_cost_breakdowns[sku] = {
                    'cost_breakdown': {
                        'ordering_cost': 0,
                        'purchase_cost': 0,
                        'holding_cost': 0,
                        'stockout_cost': 0,
                        'waste_cost': 0,
                        'total_cost': 0
                    },
                    'cost_percentages': {
                        'ordering_cost': 0,
                        'purchase_cost': 0,
                        'holding_cost': 0,
                        'stockout_cost': 0,
                        'waste_cost': 0
                    },
                    'iterations': 0
                }

        # Create the summary with average costs per implemented strategy
        summary = {
            "total_skus": len([sku for sku, runs in all_optimizations.items() if runs]),
            "simulation_days": len(next(iter(all_optimizations.values()))[0]['result']['optimized_strategy']['purchase_schedule']) if all_optimizations else 0,
            "optimized_total_cost": total_costs['total_cost'],
            "avg_service_level": total_service_level / total_iterations if total_iterations > 0 else 0,
            "total_savings": total_savings,
            "cost_breakdown": total_costs,
            "cost_percentages": cost_percentages,
            "total_iterations": total_iterations,
            "num_strategy_iterations": num_iterations,  # Number of complete strategies analyzed
            "is_average_cost": True  # Flag to indicate that these are average costs, not totals
        }


        return render_template(
            "optimize.html",
            all_optimizations=all_optimizations,
            summary=summary,
            sku_cost_breakdowns=sku_cost_breakdowns,
            all_available_skus=all_available_skus,
            results=all_optimizations
        )

    except Exception as e:
        print(f"Optimization error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return render_template("optimize.html", error=f"Error optimizing purchase schedule: {str(e)}")


def optimize_purchase_schedule(sku, demands, lead_times, initial_stock, unit_cost, margin,
                             shipping_cost, holding_cost_rate, stockout_penalty_multiplier, simulation_days,
                             expiration_days=365, waste_cost_rate=80.0):
    """
    Optimizes when and how much to order for a SKU using Integer Linear Programming (ILP).
    Returns the optimal purchase plan, minimizing total costs.
    """
    # Limit simulation days to avoid memory problems
    max_simulation_days = min(simulation_days, 90)  # Maximum 90 days to avoid OOM
    
    # Create optimization problem
    problem = LpProblem(f"Inventory_Optimization_{sku}", LpMinimize)
    
    # Parameters
    periods = range(max_simulation_days)
    max_lead_time = max(lead_times) if lead_times else 5
    
    # Adjust demands and lead_times to day limit
    demands = demands[:max_simulation_days]
    if len(lead_times) > max_simulation_days:
        lead_times = lead_times[:max_simulation_days]
    
    # Stockout penalty value - use margin as opportunity cost
    stockout_cost = margin * stockout_penalty_multiplier
    
    # Waste cost per expired unit
    waste_cost_per_unit = unit_cost * (waste_cost_rate / 100.0)
    
    # Decision variables
    # x[t]: Quantity to order in period t
    x = {t: LpVariable(f"order_{t}", lowBound=0, cat=LpInteger) for t in periods}
    
    # i[t]: Inventory level at end of period t
    i = {t: LpVariable(f"inventory_{t}", lowBound=0) for t in periods}
    
    # s[t]: Unmet demand in period t (shortage/stockout)
    s = {t: LpVariable(f"shortage_{t}", lowBound=0) for t in periods}
    
    # y[t]: Binary variable, 1 if order is placed in period t, 0 if not
    y = {t: LpVariable(f"order_placed_{t}", cat='Binary') for t in periods}
    
    # w[t]: Quantity of product that expires in period t (waste)
    w = {t: LpVariable(f"waste_{t}", lowBound=0) for t in periods}
    
    # Objective function: minimize total cost
    total_cost = (
        # Purchase cost
        lpSum(x[t] * unit_cost for t in periods) +
        # Fixed cost per order
        lpSum(y[t] * shipping_cost for t in periods) +
        # Storage cost
        lpSum(i[t] * unit_cost * holding_cost_rate for t in periods) +
        # Penalty for unmet demand (use margin as opportunity cost)
        lpSum(s[t] * stockout_cost for t in periods) +
        # Cost for waste of expired products
        lpSum(w[t] * waste_cost_per_unit for t in periods)
    )
    problem += total_cost
    
    # Constraints
    
    # Inventory balance
    for t in periods:
        # Orders arriving in period t
        incoming_orders = lpSum(x[t_prime] for t_prime in periods 
                              if t_prime + lead_times[min(t_prime, len(lead_times)-1)] == t)
        
        if t == 0:
            # Initial balance
            problem += i[t] == initial_stock + incoming_orders - demands[t] + s[t]
        else:
            # Balance in subsequent periods
            problem += i[t] == i[t-1] + incoming_orders - demands[t] + s[t]
    
    # Minimum order size constraint (100 units)
    min_order_size = 100
    for t in periods:
        problem += x[t] >= min_order_size * y[t]
        problem += x[t] <= 10000 * y[t]  # Arbitrarily large upper limit
    
    # Minimum service level constraint (at least 85%)
    total_demand = sum(demands)
    total_shortage = lpSum(s[t] for t in periods)
    problem += total_shortage <= 0.15 * total_demand  # Maximum 15% unmet demand
    
    # Simplified waste constraints
    # Approximation: products arriving in period t may expire after expiration_days
    for t in periods:
        if t + expiration_days < simulation_days:
            # Part of inventory ordered in t may expire
            expiry_factor = max(0, min(1, (simulation_days - t - expiration_days) / simulation_days))
            problem += w[t] >= x[t] * expiry_factor * 0.1  # 10% minimum of products may expire
        else:
            problem += w[t] == 0  # Not enough time to expire
    
    # Solve the problem with time limit and optimality gap
    solver = pulp.PULP_CBC_CMD(timeLimit=45, gapRel=0.05)  # 45 seconds maximum, 5% gap
    problem.solve(solver)
    
    # Verificar si se encontró una solución
    if problem.status != 1:  # 1 = Optimal
        print(f"Warning: Optimization for SKU {sku} didn't reach optimal solution. Status: {problem.status}")
    
    # Extract results
    purchase_schedule = []
    for t in periods:
        purchase_schedule.append({
            'day': t,
            'quantity': int(x[t].value()) if x[t].value() is not None else 0,
            'delivery_day': t + lead_times[min(t, len(lead_times)-1)] if x[t].value() > 0 else None
        })
    
    # Analyze optimized strategy performance
    stockout_days = sum(1 for t in periods if s[t].value() > 0)
    service_level = 100 * (1 - stockout_days / len(periods))
    
    # Calculate detailed costs
    ordering_cost = sum(shipping_cost * y[t].value() for t in periods)
    purchase_cost = sum(unit_cost * x[t].value() for t in periods)
    holding_cost = sum(unit_cost * holding_cost_rate * i[t].value() for t in periods)
    stockout_cost_total = sum(stockout_cost * s[t].value() for t in periods)
    waste_cost_total = sum(waste_cost_per_unit * w[t].value() for t in periods)
    
    return {
        'sku': sku,
        'product_name': 'Product',  # You can get this data from the database
        'optimized_strategy': {
            'purchase_schedule': purchase_schedule,
            'total_cost': ordering_cost + purchase_cost + holding_cost + stockout_cost_total + waste_cost_total,
            'service_level': service_level,
            'total_orders': sum(1 for day in purchase_schedule if day['quantity'] > 0),
            'total_quantity_ordered': sum(day['quantity'] for day in purchase_schedule),
            'average_inventory': sum(i[t].value() for t in periods) / len(periods),
            'stockout_days': stockout_days,
            'cost_breakdown': {
                'ordering_cost': ordering_cost,
                'purchase_cost': purchase_cost,
                'holding_cost': holding_cost,
                'stockout_cost': stockout_cost_total,
                'waste_cost': waste_cost_total
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
    '''Generate just-in-time strategy: order exactly what\'s needed.'''
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
                          margin, shipping_cost, holding_cost_rate, stockout_penalty_multiplier,
                          expiration_days=365, waste_cost_rate=80.0):
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

        # Calculate storage cost
        holding_cost = current_stock * unit_cost * holding_cost_rate
        total_holding_cost += holding_cost
        total_cost += holding_cost

        # Satisfy demand
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

    # Calculate metrics
    avg_inventory = sum(inventory_levels) / len(inventory_levels) if inventory_levels else 0
    service_level = ((len(demands) - stockout_days) / len(demands)) * 100 if demands else 100

    # Calculate baseline cost (without optimization)
    baseline_cost = sum(demands) * unit_cost * 1.2  # Assumes 20% more cost without optimization

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
    # Get product information
    products = execute_query("SELECT SKU, NAME, MARGIN, COST FROM products")
    product_info = {}
    for prod in products:
        # Use the safe function to get complete product info including EXPIRATION
        product_info[str(prod['SKU'])] = get_product_info_safe(prod['SKU'])
    all_skus = [str(prod['SKU']) for prod in products]

    # Check if user clicked "Show solution" button
    if request.method == "POST" and request.form.get('action') == 'show_solution':
        # --- REBUILD best_strategies same as in /optimize ---
        # Get the maximum iteration number from the database
        max_iteration_result = execute_query("SELECT MAX(Iteration) as max_iter FROM simulation")
        max_iterations = max_iteration_result[0]['max_iter'] if max_iteration_result and max_iteration_result[0]['max_iter'] else 10

        best_strategies = {}
        for sku in all_skus:
            # Get all available simulations for this SKU
            scenarios = []
            strategies = []
            for iteration in range(1, max_iterations + 1):
                sim_rows = execute_query(
                    "SELECT * FROM simulation WHERE SKU = ? AND Iteration = ?",
                    sku, iteration
                )
                if not sim_rows:
                    continue
                demands = [int(row['DEMAND']) for row in sim_rows]
                lead_times = [int(row['LEAD_TIME']) for row in sim_rows]
                initial_stock = float(sim_rows[0]['STOCK']) + int(sim_rows[0]['DEMAND']) if sim_rows else 0
                unit_cost = float(product_info[sku].get('COST', 10))
                margin = float(product_info[sku].get('MARGIN', 0))
                expiration_days = int(product_info[sku].get('EXPIRATION', 365))
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
                    simulation_days=len(demands),
                    expiration_days=expiration_days,
                    waste_cost_rate=80.0
                )
                strategies.append(result['optimized_strategy']['purchase_schedule'])
                scenarios.append({
                    "demands": demands,
                    "lead_times": lead_times,
                    "initial_stock": initial_stock
                })

            # Evaluate each strategy in all scenarios
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
                        2.0,
                        expiration_days,
                        80.0
                    )
                avg_cost = total_cost / len(scenarios) if scenarios else float('inf')
                avg_costs.append(avg_cost)

            # Select the best strategy
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
            calendar_matrix=calendar_matrix,
            show_solution=True
        )
    else:
        # Just show the page without solution
        return render_template(
            "results.html",
            best_strategies={},
            product_info=product_info,
            sku_list=[],
            calendar_matrix=[],
            show_solution=False
        )

# ============================================================================
# ANALYTICS & REPORTING ROUTES
# ============================================================================

@app.route("/analytics")
def analytics():
    """Advanced analytics and reporting dashboard"""
    return render_template("analytics.html")

@app.route("/api/analytics/overview")
def analytics_overview():
    """Get overview analytics data"""
    try:
        # Get basic inventory stats
        inventory_stats = execute_query(
            "SELECT COUNT(DISTINCT SKU) as total_skus, SUM(VOLUME) as total_volume, AVG(VOLUME) as avg_volume_per_sku FROM inventory"
        )
        
        # Get transaction stats
        transaction_stats = execute_query(
            "SELECT COUNT(*) as total_transactions, SUM(VOLUME) as total_volume_moved, COUNT(DISTINCT SKU) as active_skus FROM transactions"
        )
        
        # Get top performing SKUs
        top_skus = execute_query(
            "SELECT SKU, SUM(VOLUME) as total_volume, COUNT(*) as transaction_count FROM transactions GROUP BY SKU ORDER BY total_volume DESC LIMIT 10"
        )
        
        # Calculate inventory turnover - aggregate inventory by SKU first to handle duplicates
        turnover_data = execute_query(
            """SELECT inv.SKU, 
                      inv.current_stock, 
                      COALESCE(t.total_moved, 0) as total_moved, 
                      CASE 
                          WHEN inv.current_stock > 0 THEN ROUND(COALESCE(t.total_moved, 0) / inv.current_stock, 2)
                          WHEN COALESCE(t.total_moved, 0) > 0 THEN 999.99
                          ELSE 0 
                      END as turnover_ratio 
               FROM (SELECT SKU, SUM(VOLUME) as current_stock FROM inventory GROUP BY SKU) inv
               LEFT JOIN (SELECT SKU, SUM(ABS(VOLUME)) as total_moved FROM transactions GROUP BY SKU) t 
               ON inv.SKU = t.SKU 
               WHERE COALESCE(t.total_moved, 0) > 0 OR inv.current_stock > 0
               ORDER BY turnover_ratio DESC"""
        )
        
        # Debug logging
        print(f"DEBUG - Total turnover_data records: {len(turnover_data)}")
        if turnover_data:
            print(f"DEBUG - SKUs in turnover_data: {[row['SKU'] for row in turnover_data]}")
            print(f"DEBUG - First 5 turnover records:")
            for row in turnover_data[:5]:
                print(f"  SKU {row['SKU']}: stock={row['current_stock']}, moved={row['total_moved']}, ratio={row['turnover_ratio']}")
        
        return jsonify({
            "inventory_stats": inventory_stats[0] if inventory_stats else {},
            "transaction_stats": transaction_stats[0] if transaction_stats else {},
            "top_skus": top_skus,
            "turnover_data": turnover_data[:10]  # Top 10 by turnover
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/analytics/trends")
def analytics_trends():
    """Get trend analysis data"""
    try:
        # Get SKU performance trends
        sku_trends = execute_query(
            "SELECT SKU, SUM(CASE WHEN VOLUME > 0 THEN VOLUME ELSE 0 END) as inbound, SUM(CASE WHEN VOLUME < 0 THEN ABS(VOLUME) ELSE 0 END) as outbound, COUNT(*) as activity_count FROM transactions GROUP BY SKU HAVING activity_count >= 1 ORDER BY activity_count DESC, inbound DESC LIMIT 20"
        )
        
        # Calculate demand variability
        demand_analysis = []
        for sku_data in sku_trends:
            sku = sku_data['SKU']
            # Include all transaction types to capture full demand picture
            print(f"DEBUG - Querying volumes for SKU: {sku}")
            volumes = execute_query(
                "SELECT VOLUME FROM transactions WHERE SKU = ?", 
                sku
            )
            print(f"DEBUG - Found {len(volumes)} volumes for SKU {sku}")
            if len(volumes) >= 1:
                vol_values = [v['VOLUME'] for v in volumes]  # Use signed values for net volume
                abs_vol_values = [abs(v['VOLUME']) for v in volumes]  # Use absolute for variability calculation
                mean_vol = float(np.mean(abs_vol_values))
                std_vol = float(np.std(abs_vol_values)) if len(abs_vol_values) > 1 else 0.0
                cv = (std_vol / mean_vol) if mean_vol > 0 else 0.0
                
                # Get additional metrics
                total_volume = sum(vol_values)  # Net volume (with sign)
                transaction_count = len(volumes)
                
                demand_analysis.append({
                    'SKU': str(sku),
                    'mean_demand': round(mean_vol, 2),
                    'std_demand': round(std_vol, 2),
                    'coefficient_variation': round(cv, 3),
                    'demand_pattern': 'Stable' if cv < 0.3 else 'Variable' if cv < 0.7 else 'Highly Variable',
                    'total_volume': total_volume,
                    'transaction_count': transaction_count
                })
        
        # Debug logging
        print(f"DEBUG - Total demand_analysis records: {len(demand_analysis)}")
        if demand_analysis:
            print(f"DEBUG - SKUs in demand_analysis: {[row['SKU'] for row in demand_analysis[:15]]}")
        
        return jsonify({
            "sku_trends": sku_trends,
            "demand_analysis": demand_analysis
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/analytics/forecast/<sku>")
def analytics_forecast(sku):
    """Generate demand forecast for a specific SKU"""
    try:
        # Get historical demand data (only demand-side transactions)
        historical_data = execute_query("""
            SELECT ABS(VOLUME) as demand 
            FROM transactions 
            WHERE SKU = ? AND DOCUMENT_TYPE IN ('Invoices', 'Debit Notes') AND VOLUME != 0
            ORDER BY rowid
        """, sku)
        
        if len(historical_data) < 5:
            return jsonify({"error": "Insufficient data for forecasting"}), 400
        
        # Prepare data for forecasting
        demands = [d['demand'] for d in historical_data]
        X = np.array(range(len(demands))).reshape(-1, 1)
        y = np.array(demands)
        
        # Simple linear regression forecast
        model = LinearRegression()
        model.fit(X, y)
        
        # Forecast next 30 periods
        future_periods = 30
        future_X = np.array(range(len(demands), len(demands) + future_periods)).reshape(-1, 1)
        forecast = model.predict(future_X)
        
        # Calculate trend and seasonality
        trend = model.coef_[0]
        trend_direction = "Increasing" if trend > 0.1 else "Decreasing" if trend < -0.1 else "Stable"
        
        # Calculate forecast accuracy metrics
        predicted_historical = model.predict(X)
        mae = np.mean(np.abs(y - predicted_historical))
        mse = np.mean((y - predicted_historical) ** 2)
        rmse = np.sqrt(mse)
        
        return jsonify({
            "sku": sku,
            "historical_demand": demands,
            "forecast": forecast.tolist(),
            "trend_direction": trend_direction,
            "trend_value": round(trend, 3),
            "accuracy_metrics": {
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "periods_forecasted": future_periods
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================================
# ALERTS & NOTIFICATIONS SYSTEM
# ============================================================================

@app.route("/alerts")
def alerts_dashboard():
    """Alerts and notifications dashboard"""
    response = make_response(render_template("alerts.html"))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/api/alerts/check")
def check_alerts():
    """Check for all types of alerts"""
    try:
        alerts = []

        # Low Stock Alerts
        low_stock_alerts = check_low_stock_alerts()
        alerts.extend(low_stock_alerts)

        # Stock Breaks (failed transaction attempts / stockouts)
        stock_break_alerts = check_stock_breaks()
        alerts.extend(stock_break_alerts)

        # Stagnated Items (demand < 80% of expected baseline)
        stagnated_alerts = check_stagnated_items()
        alerts.extend(stagnated_alerts)

        # Sort alerts by severity (critical, warning, info)
        severity_order = {'critical': 0, 'warning': 1, 'info': 2}
        alerts.sort(key=lambda x: severity_order.get(x['severity'], 3))
        
        return jsonify({
            "alerts": alerts,
            "total_alerts": len(alerts),
            "critical_count": len([a for a in alerts if a['severity'] == 'critical']),
            "warning_count": len([a for a in alerts if a['severity'] == 'warning']),
            "info_count": len([a for a in alerts if a['severity'] == 'info'])
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def check_low_stock_alerts():
    """Check for low stock conditions (SKU-level) using agreed rules:

    - Use 30-day rolling avg demand (compute_avg_daily_demand)
    - Days of supply = total_stock / avg_daily_demand
    - Low stock if days_of_supply <= (10 + lead_time_days)
    - Require at least 90 days of history (sku_has_min_history)
    - Severity: 'critical' if days_of_supply <= lead_time_days, else 'warning'
    """
    alerts = []

    try:
        # Gather candidate SKUs from inventory or products
        skus_rows = execute_query("SELECT DISTINCT SKU FROM inventory UNION SELECT DISTINCT SKU FROM products")
        skus = [r['SKU'] for r in skus_rows if r.get('SKU')]

        for sku in skus:
            try:
                # Skip SKUs with insufficient history
                if not sku_has_min_history(sku, min_days=90):
                    continue

                total_stock = compute_total_stock(sku)
                avg_daily = compute_avg_daily_demand(sku, days=30)

                # If there's no recent demand, skip low-stock (no consumption)
                if avg_daily <= 0:
                    continue

                days_of_supply = float(total_stock) / float(avg_daily) if avg_daily > 0 else float('inf')
                lead_time = get_supplier_lead_time(sku, default_lead=7)
                threshold_days = 10 + int(lead_time)

                if days_of_supply <= threshold_days:
                    severity = 'critical' if days_of_supply <= lead_time else 'warning'
                    message = f"Low stock: {days_of_supply:.1f} days of supply (threshold {threshold_days} days)"

                    alerts.append({
                        'type': 'low_stock',
                        'severity': severity,
                        'sku': sku,
                        'product_name': get_product_info_safe(sku).get('NAME'),
                        'message': message,
                        'details': {
                            'total_stock': total_stock,
                            'avg_daily_demand_30': round(avg_daily, 3),
                            'days_of_supply': round(days_of_supply, 2),
                            'supplier_lead_time_days': int(lead_time),
                            'threshold_days': int(threshold_days)
                        },
                        'timestamp': datetime.now().isoformat()
                    })

            except Exception as inner_e:
                print(f"Error evaluating low stock for {sku}: {inner_e}", flush=True)
                continue

        return alerts

    except Exception as e:
        print(f"Error in check_low_stock_alerts: {e}", flush=True)
        return []

def check_expiration_alerts():
    """Check for batches approaching expiration"""
    alerts = []
    
    # Get individual batches with their dates and expiration info
    batch_data = execute_query(
        "SELECT i.SKU, i.VOLUME, i.DATE as batch_date, p.EXPIRATION as expiration_days FROM inventory i LEFT JOIN products p ON i.SKU = p.SKU WHERE i.VOLUME > 0 AND i.DATE IS NOT NULL"
    )
    
    current_date = datetime.now().date()
    
    for batch in batch_data:
        sku = batch['SKU']
        volume = batch['VOLUME']
        batch_date_str = batch['batch_date']
        expiration_days = batch['expiration_days'] or 365
        
        # Parse batch date
        try:
            if isinstance(batch_date_str, str):
                batch_date = datetime.strptime(batch_date_str, '%Y-%m-%d').date()
            else:
                batch_date = batch_date_str
        except:
            continue  # Skip if date parsing fails
        
        # Calculate expiration date and days until expiration
        expiration_date = batch_date + timedelta(days=expiration_days)
        days_until_expiration = (expiration_date - current_date).days
        
        if days_until_expiration < 0:
            severity = 'critical'
            message = f"Batch expired {abs(days_until_expiration)} days ago ({volume} units)"
        elif days_until_expiration < 7:
            severity = 'critical'
            message = f"Batch expires in {days_until_expiration} days ({volume} units)"
        elif days_until_expiration < 30:
            severity = 'warning'
            message = f"Batch expires in {days_until_expiration} days ({volume} units)"
        elif days_until_expiration < 60:
            severity = 'info'
            message = f"Batch expires in {days_until_expiration} days ({volume} units)"
        else:
            continue
            
        alerts.append({
            'type': 'expiration',
            'severity': severity,
            'sku': sku,
            'message': message,
            'details': {
                'batch_volume': volume,
                'batch_date': batch_date_str,
                'expiration_date': expiration_date.isoformat(),
                'days_until_expiration': days_until_expiration,
                'expiration_period': expiration_days
            },
            'timestamp': datetime.now().isoformat()
        })
    
    return alerts

def check_reorder_point_alerts():
    """Check for reorder point alerts based on demand patterns"""
    alerts = []
    
    # Get SKUs with transaction history
    active_skus = execute_query("""
        SELECT SKU, AVG(ABS(VOLUME)) as avg_demand,
               COUNT(*) as transaction_count
        FROM transactions
        GROUP BY SKU
        HAVING transaction_count >= 3
    """)
    
    for sku_data in active_skus:
        sku = sku_data['SKU']
        avg_demand = sku_data['avg_demand']
        
        # Get current inventory
        current_inventory = execute_query("""
            SELECT COALESCE(SUM(VOLUME), 0) as stock 
            FROM inventory WHERE SKU = ?
        """, sku)
        
        current_stock = current_inventory[0]['stock'] if current_inventory else 0
        
        # Calculate reorder point (safety stock + lead time demand)
        lead_time_days = 7  # Assume 7 day lead time
        safety_stock = avg_demand * 3  # 3 days safety stock
        reorder_point = (avg_demand * lead_time_days) + safety_stock
        
        if current_stock <= reorder_point:
            severity = 'warning' if current_stock > reorder_point * 0.5 else 'critical'
            message = f"Reorder recommended: Stock below reorder point ({reorder_point:.0f})"
            
            alerts.append({
                'type': 'reorder_point',
                'severity': severity,
                'sku': sku,
                'message': message,
                'details': {
                    'current_stock': current_stock,
                    'reorder_point': round(reorder_point, 0),
                    'avg_demand': round(avg_demand, 2),
                    'recommended_order_qty': round(avg_demand * 30, 0)  # 30 days supply
                },
                'timestamp': datetime.now().isoformat()
            })
    
    return alerts

def check_demand_variance_alerts():
    """Check for unusual demand patterns"""
    alerts = []
    
    # Get SKUs with sufficient history
    skus_with_history = execute_query("""
        SELECT SKU, COUNT(*) as transaction_count
        FROM transactions
        GROUP BY SKU
        HAVING transaction_count >= 5
    """)
    
    for sku_data in skus_with_history:
        sku = sku_data['SKU']
        
        # Get recent demand data
        volumes = execute_query("""
            SELECT ABS(VOLUME) as demand 
            FROM transactions 
            WHERE SKU = ?
            ORDER BY rowid DESC
            LIMIT 10
        """, sku)
        
        if len(volumes) < 5:
            continue
            
        demands = [v['demand'] for v in volumes]
        mean_demand = np.mean(demands)
        std_demand = np.std(demands)
        cv = std_demand / mean_demand if mean_demand > 0 else 0
        
        # Check for high variability
        if cv > 1.0:  # Coefficient of variation > 100%
            severity = 'warning'
            message = f"High demand variability detected (CV: {cv:.2f})"
            
            alerts.append({
                'type': 'demand_variance',
                'severity': severity,
                'sku': sku,
                'message': message,
                'details': {
                    'coefficient_variation': round(cv, 3),
                    'mean_demand': round(mean_demand, 2),
                    'std_demand': round(std_demand, 2),
                    'pattern': 'Highly Variable'
                },
                'timestamp': datetime.now().isoformat()
            })
    
    return alerts


def check_stagnated_items(recent_days=30, baseline_days=90, stagnation_ratio=0.8, min_history_days=90, limit=100, offset=0):
    """Detect stagnated items: recent demand is less than `stagnation_ratio` of expected demand.

    - recent_days: window to measure recent demand (default 30)
    - baseline_days: period to compute expected average demand (default 90)
    - stagnation_ratio: threshold (default 0.8)
    - min_history_days: require at least this much history (default 90)
    """
    alerts = []
    try:
        # Build candidate SKU list
        skus_rows = execute_query("SELECT DISTINCT SKU FROM transactions UNION SELECT DISTINCT SKU FROM inventory UNION SELECT DISTINCT SKU FROM products")
        skus = [r['SKU'] for r in skus_rows if r.get('SKU')]

        for sku in skus:
            try:
                # Require minimum history
                if not sku_has_min_history(sku, min_history_days):
                    continue

                # recent demand (last recent_days)
                recent_q = execute_query(
                    "SELECT COALESCE(SUM(ABS(VOLUME)),0) as demand_30 FROM transactions WHERE SKU = ? AND DOCUMENT_TYPE IN ('Invoices', 'Debit Notes', 'Credit Notes') AND DATE >= date('now', ?)",
                    sku, f"-{int(recent_days)} days")
                recent_demand = float(recent_q[0]['demand_30']) if recent_q and recent_q[0] and recent_q[0].get('demand_30') is not None else 0.0

                # baseline demand (last baseline_days)
                baseline_q = execute_query(
                    "SELECT COALESCE(SUM(ABS(VOLUME)),0) as demand_90 FROM transactions WHERE SKU = ? AND DOCUMENT_TYPE IN ('Invoices', 'Debit Notes', 'Credit Notes') AND DATE >= date('now', ?)",
                    sku, f"-{int(baseline_days)} days")
                baseline_demand = float(baseline_q[0]['demand_90']) if baseline_q and baseline_q[0] and baseline_q[0].get('demand_90') is not None else 0.0

                if baseline_demand <= 0:
                    continue  # no baseline to compare

                # expected demand for recent_days based on baseline average
                avg_daily_expected = baseline_demand / float(baseline_days)
                expected_recent = avg_daily_expected * float(recent_days)

                # Avoid division by zero
                if expected_recent <= 0:
                    continue

                ratio = recent_demand / expected_recent if expected_recent > 0 else 0.0

                if ratio < float(stagnation_ratio):
                    # lookup last sale date for context
                    last_q = execute_query("SELECT MAX(DATE) as last_sale FROM transactions WHERE SKU = ? AND DOCUMENT_TYPE IN ('Invoices', 'Debit Notes', 'Credit Notes')", sku)
                    last_sale = last_q[0]['last_sale'] if last_q and last_q[0] else None

                    alerts.append({
                        'type': 'stagnated',
                        'severity': 'info',
                        'sku': sku,
                        'product_name': get_product_info_safe(sku).get('NAME'),
                        'message': f"Stagnated: recent demand {int(recent_demand)} < {int(stagnation_ratio*100)}% of expected ({int(expected_recent)})",
                        'details': {
                            'recent_demand': recent_demand,
                            'expected_recent': round(expected_recent, 2),
                            'baseline_days': baseline_days,
                            'recent_days': recent_days,
                            'ratio': round(ratio, 3),
                            'last_sale_date': last_sale
                        },
                        'timestamp': datetime.now().isoformat()
                    })

            except Exception as inner_e:
                print(f"Error evaluating stagnation for {sku}: {inner_e}", flush=True)
                continue

        return alerts

    except Exception as e:
        print(f"Error in check_stagnated_items: {e}", flush=True)
        return []


def check_stock_breaks(window_days=30, min_history_days=90, limit=100, offset=0):
    """Detect stock breaks: failed transaction attempts due to insufficient stock.

    Strategy:
    - If the `transactions` table contains an explicit success/failure column or failure reason, use it.
    - Otherwise, return empty list and log that explicit failure data is required for reliable detection.
    """
    alerts = []
    try:
        # Inspect transactions table for helpful columns
        try:
            cols = execute_query("PRAGMA table_info(transactions)")
            col_names = [c['name'].upper() for c in cols]
        except Exception:
            col_names = []

        # Prefer explicit columns if present
        if 'SUCCESS' in col_names or 'FAILURE_REASON' in col_names or 'STATUS' in col_names:
            # Build query dynamically depending on available columns
            where_clauses = []
            params = []
            date_clause = "DATE >= date('now', ?)"
            params.append(f"-{int(window_days)} days")

            # If SUCCESS column exists, look for success=0
            if 'SUCCESS' in col_names:
                where_clauses.append("SUCCESS = 0")

            # If FAILURE_REASON exists, look for keywords indicating insufficient stock
            if 'FAILURE_REASON' in col_names:
                where_clauses.append("(FAILURE_REASON LIKE '%stock%' OR FAILURE_REASON LIKE '%insufficient%')")

            # If STATUS exists, look for 'FAILED' or similar
            if 'STATUS' in col_names:
                where_clauses.append("(STATUS = 'FAILED' OR STATUS = 'FAIL')")

            if not where_clauses:
                return []

            combined_where = " AND (" + " OR ".join(where_clauses) + ") AND " + date_clause

            query = f"SELECT SKU, COUNT(*) as failed_attempts, MAX(DATE) as last_failed_at, COUNT(DISTINCT DATE) as distinct_failed_days FROM transactions WHERE {combined_where} GROUP BY SKU ORDER BY failed_attempts DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            rows = execute_query(query, *params)

            for r in rows:
                sku = r['SKU']
                if not sku_has_min_history(sku, min_history_days):
                    continue
                alerts.append({
                    'type': 'stock_break',
                    'severity': 'critical',
                    'sku': sku,
                    'product_name': get_product_info_safe(sku).get('NAME'),
                    'message': f"Stock break: {int(r['failed_attempts'])} failed attempts in last {window_days} days",
                    'details': {
                        'failed_attempts': int(r['failed_attempts']),
                        'last_failed_at': r.get('last_failed_at'),
                        'distinct_failed_days': int(r.get('distinct_failed_days') or 0)
                    },
                    'timestamp': datetime.now().isoformat()
                })

            return alerts

        else:
            print("Stock break detection skipped: transactions table has no explicit success/failure columns. Add 'SUCCESS' or 'FAILURE_REASON' to enable.", flush=True)
            return []

    except Exception as e:
        print(f"Error checking stock breaks: {e}", flush=True)
        return []


# --- Alerts API endpoints (summary + per-detector drilldowns with pagination and CSV export) ---
def _list_to_csv_response(rows, filename="export.csv"):
    """Convert a list of dicts (alerts) to a CSV download response."""
    try:
        if not rows:
            # Create empty dataframe with common columns
            df = pd.DataFrame(columns=['type', 'severity', 'sku', 'message', 'timestamp'])
        else:
            # Normalize nested 'details' into columns
            df = pd.json_normalize(rows)
            # Prefer to present main fields first
            cols = []
            for c in ['type', 'severity', 'sku', 'message', 'timestamp']:
                if c in df.columns:
                    cols.append(c)
            # Append remaining columns
            remaining = [c for c in df.columns if c not in cols]
            df = df[cols + remaining]

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        resp = make_response(output.getvalue())
        resp.headers["Content-Disposition"] = f"attachment; filename={filename}"
        resp.headers["Content-type"] = "text/csv"
        return resp
    except Exception as e:
        return jsonify({"error": f"CSV export failed: {e}"}), 500


@app.route('/api/alerts/summary')
@rate_limited()
def api_alerts_summary():
    """Return counts for the three alert categories and top-N items for dashboard cards."""
    try:
        top_n = int(request.args.get('top_n', 5))

        # Use only the three main detectors for full consistency
        low = check_low_stock_alerts()
        breaks = check_stock_breaks()
        stagnated = check_stagnated_items()

        total_alerts = len(low) + len(breaks) + len(stagnated)

        summary = {
            'low_stock_count': len(low),
            'stock_break_count': len(breaks),
            'stagnated_count': len(stagnated),
            'total_alerts': total_alerts,
            'low_stock_top': low[:top_n],
            'stock_breaks_top': breaks[:top_n],
            'stagnated_top': stagnated[:top_n]
        }

        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/low_stock')
@rate_limited()
def api_alerts_low_stock():
    """Paginated drilldown for low stock alerts. Query params: limit, offset, export=csv"""
    try:
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        export = request.args.get('export', '').lower() == 'csv'

        items = check_low_stock_alerts()

        # Server-side filtering
        sku_filter = request.args.get('sku', '').strip().lower()
        name_filter = request.args.get('name', '').strip().lower()
        severity_filter = request.args.get('severity', '').strip().lower()
        sort_by = request.args.get('sort_by', 'days_of_supply')
        sort_dir = request.args.get('sort_dir', 'asc').lower()

        def matches(item):
            if sku_filter and sku_filter not in (item.get('sku') or '').lower():
                return False
            if name_filter and name_filter not in ((item.get('product_name') or '')).lower():
                return False
            if severity_filter and severity_filter != (item.get('severity') or '').lower():
                return False
            return True

        filtered = [i for i in items if matches(i)]

        # Sorting
        reverse = sort_dir == 'desc'
        if sort_by == 'severity':
            order = {'critical': 0, 'warning': 1, 'info': 2}
            filtered.sort(key=lambda x: order.get(x.get('severity'), 99), reverse=reverse)
        elif sort_by == 'timestamp':
            filtered.sort(key=lambda x: x.get('timestamp') or '', reverse=reverse)
        else:
            # default: try days_of_supply inside details
            filtered.sort(key=lambda x: float(x.get('details', {}).get('days_of_supply') or 99999), reverse=reverse)

        total = len(filtered)
        page = filtered[offset: offset + limit]

        if export:
            return _list_to_csv_response(page, filename='low_stock_alerts.csv')

        return jsonify({'count': total, 'items': page, 'limit': limit, 'offset': offset})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/stock_breaks')
@rate_limited()
def api_alerts_stock_breaks():
    """Paginated drilldown for stock breaks. Query params: window_days, limit, offset, export=csv"""
    try:
        window_days = int(request.args.get('window_days', 30))
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        export = request.args.get('export', '').lower() == 'csv'

        # check_stock_breaks supports window_days, limit, offset
        items = check_stock_breaks(window_days=window_days, limit=10000, offset=0)

        # Filters
        sku_filter = request.args.get('sku', '').strip().lower()
        name_filter = request.args.get('name', '').strip().lower()
        severity_filter = request.args.get('severity', '').strip().lower()
        sort_by = request.args.get('sort_by', 'failed_attempts')
        sort_dir = request.args.get('sort_dir', 'desc').lower()

        def matches(item):
            if sku_filter and sku_filter not in (item.get('sku') or '').lower():
                return False
            if name_filter and name_filter not in ((item.get('product_name') or '')).lower():
                return False
            if severity_filter and severity_filter != (item.get('severity') or '').lower():
                return False
            return True

        filtered = [i for i in items if matches(i)]

        reverse = sort_dir == 'desc'
        if sort_by == 'last_failed_at':
            filtered.sort(key=lambda x: x.get('details', {}).get('last_failed_at') or '', reverse=reverse)
        else:
            # default: failed_attempts
            filtered.sort(key=lambda x: int(x.get('details', {}).get('failed_attempts') or 0), reverse=reverse)

        total = len(filtered)
        page = filtered[offset: offset + limit]

        if export:
            return _list_to_csv_response(page, filename='stock_breaks_alerts.csv')

        return jsonify({'count': total, 'items': page, 'limit': limit, 'offset': offset, 'window_days': window_days})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/stagnated')
@rate_limited()
def api_alerts_stagnated():
    """Paginated drilldown for stagnated items. Query params: recent_days, baseline_days, limit, offset, export=csv"""
    try:
        recent_days = int(request.args.get('recent_days', 30))
        baseline_days = int(request.args.get('baseline_days', 90))
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        export = request.args.get('export', '').lower() == 'csv'

        items = check_stagnated_items(recent_days=recent_days, baseline_days=baseline_days, min_history_days=90, limit=10000, offset=0)

        sku_filter = request.args.get('sku', '').strip().lower()
        name_filter = request.args.get('name', '').strip().lower()
        severity_filter = request.args.get('severity', '').strip().lower()
        sort_by = request.args.get('sort_by', 'ratio')
        sort_dir = request.args.get('sort_dir', 'asc').lower()

        def matches(item):
            if sku_filter and sku_filter not in (item.get('sku') or '').lower():
                return False
            if name_filter and name_filter not in ((item.get('product_name') or '')).lower():
                return False
            if severity_filter and severity_filter != (item.get('severity') or '').lower():
                return False
            return True

        filtered = [i for i in items if matches(i)]

        reverse = sort_dir == 'desc'
        if sort_by == 'timestamp':
            filtered.sort(key=lambda x: x.get('timestamp') or '', reverse=reverse)
        else:
            filtered.sort(key=lambda x: float(x.get('details', {}).get('ratio') or 0.0), reverse=reverse)

        total = len(filtered)
        page = filtered[offset: offset + limit]

        if export:
            return _list_to_csv_response(page, filename='stagnated_alerts.csv')

        return jsonify({'count': total, 'items': page, 'limit': limit, 'offset': offset, 'recent_days': recent_days, 'baseline_days': baseline_days})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# BARCODE/QR CODE INTEGRATION
# ============================================================================

@app.route("/barcode")
def barcode_scanner():
    """Barcode scanning and management interface"""
    return render_template("barcode.html")

@app.route("/api/barcode/generate/<sku>")
def generate_barcode(sku):
    """Generate barcode for a SKU"""
    try:
        import qrcode
        from io import BytesIO
        import base64
        
        # Get product info
        product_info = get_product_info_safe(sku)
        
        # Create QR code data
        qr_data = {
            "sku": sku,
            "name": product_info.get("NAME", "Unknown"),
            "cost": product_info.get("COST", 0),
            "margin": product_info.get("MARGIN", 0)
        }
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(str(qr_data))
        qr.make(fit=True)
        
        # Create QR code image
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            "sku": sku,
            "qr_code": f"data:image/png;base64,{img_base64}",
            "product_info": product_info
        })
        
    except ImportError:
        return jsonify({"error": "QR code library not installed. Run: pip install qrcode[pil]"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/barcode/lookup", methods=["POST"])
def barcode_lookup():
    """Look up product by barcode/QR code data"""
    try:
        data = request.get_json()
        barcode_data = data.get('barcode_data', '')
        
        if not barcode_data:
            return jsonify({"error": "No barcode data provided"}), 400
        
        # Try to parse as JSON (QR code)
        try:
            import json
            parsed_data = json.loads(barcode_data.replace("'", '"'))
            if isinstance(parsed_data, dict) and 'sku' in parsed_data:
                sku = parsed_data['sku']
            else:
                sku = str(barcode_data)
        except:
            # Treat as plain SKU
            sku = str(barcode_data)
        
        # Get product information
        product_info = get_product_info_safe(sku)
        
        # Get current inventory (aggregate in case of duplicates)
        inventory = execute_query("SELECT SUM(VOLUME) as total_stock FROM inventory WHERE SKU = ?", sku)
        current_stock = inventory[0]['total_stock'] if inventory and inventory[0]['total_stock'] else 0
        
        # Get recent transactions
        recent_transactions = execute_query("""
            SELECT * FROM transactions 
            WHERE SKU = ? 
            ORDER BY rowid DESC 
            LIMIT 5
        """, sku)
        
        return jsonify({
            "sku": sku,
            "found": True,
            "product_info": product_info,
            "current_stock": current_stock,
            "recent_transactions": recent_transactions
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/barcode/transaction", methods=["POST"])
def barcode_transaction():
    """Process a transaction via barcode scan"""
    try:
        data = request.get_json()
        sku = data.get('sku')
        transaction_type = data.get('type')  # 'in' or 'out'
        volume = float(data.get('volume', 0))
        
        if not sku or not transaction_type or volume <= 0:
            return jsonify({"error": "Missing required fields"}), 400
        
        # Determine transaction document type and adjust volume
        if transaction_type == 'out':
            doc_type = 'Invoices'  # Default for outbound
            volume = -abs(volume)
        else:
            doc_type = 'Receipts'  # Default for inbound
            volume = abs(volume)
        
        # Process the transaction
        result = process_transaction(sku, volume, doc_type)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================================
# CSV/EXCEL EXPORT/IMPORT FUNCTIONALITY
# ============================================================================

@app.route("/import_export")
def import_export_interface():
    """Import/Export data management interface"""
    return render_template("import_export.html")

@app.route("/import_export_professional")
def import_export_professional():
    """Professional Import/Export data management interface - NEW DESIGN"""
    return render_template("import_export_professional.html")

@app.route("/design_comparison")
def design_comparison():
    """Compare current vs professional design side by side"""
    try:
        return render_template("simple_comparison.html")
    except Exception as e:
        return f"Error loading template: {str(e)}"

@app.route("/test")
def test_route():
    """Test route to verify Flask is working"""
    return "<h1>Flask is working! Design comparison should be available at /design_comparison</h1>"

@app.route("/api/export/inventory")
def export_inventory():
    """Export inventory data to CSV"""
    try:
        # Get inventory data with product info
        inventory_data = execute_query(
            "SELECT i.SKU, i.VOLUME as Current_Stock FROM inventory i WHERE i.VOLUME > 0 ORDER BY i.SKU"
        )
        
        # Create DataFrame
        df = pd.DataFrame(inventory_data)
        
        # Add product information
        for index, row in df.iterrows():
            sku = row['SKU']
            product_info = get_product_info_safe(sku)
            df.at[index, 'Product_Name'] = product_info.get('NAME', 'Unknown')
            df.at[index, 'Cost'] = product_info.get('COST', 0)
            df.at[index, 'Margin'] = product_info.get('MARGIN', 0)
            df.at[index, 'Expiration_Days'] = product_info.get('EXPIRATION', 365)
        
        # Reorder columns
        df = df[['SKU', 'Product_Name', 'Current_Stock', 'Cost', 'Margin', 'Expiration_Days']]
        
        # Create CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=inventory_export.csv"
        response.headers["Content-type"] = "text/csv"
        
        return response
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/export/transactions")
def export_transactions():
    """Export transaction history to CSV"""
    try:
        # Get transaction data
        transactions = execute_query(
            "SELECT SKU, VOLUME, CASE WHEN VOLUME > 0 THEN 'IN' ELSE 'OUT' END as Type, ABS(VOLUME) as Quantity, rowid as Transaction_ID FROM transactions ORDER BY rowid DESC"
        )
        
        df = pd.DataFrame(transactions)
        
        # Add timestamp (simulated)
        df['Timestamp'] = pd.date_range(
            start='2024-01-01', 
            periods=len(df), 
            freq='1H'
        ).strftime('%Y-%m-%d %H:%M:%S')
        
        # Reorder columns
        df = df[['Transaction_ID', 'SKU', 'Type', 'Quantity', 'VOLUME', 'Timestamp']]
        
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=transactions_export.csv"
        response.headers["Content-type"] = "text/csv"
        
        return response
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/export/analytics")
def export_analytics():
    """Export analytics report to CSV"""
    try:
        # Get comprehensive analytics data
        analytics_data = []
        
        # Get all SKUs with stats
        skus = execute_query("SELECT DISTINCT SKU FROM inventory")
        
        for item in skus:
            sku = item['SKU']
            
            # Get inventory
            inventory = execute_query("SELECT VOLUME FROM inventory WHERE SKU = ?", sku)
            current_stock = inventory[0]['VOLUME'] if inventory else 0
            
            # Get transaction stats
            transactions = execute_query(
                "SELECT COUNT(*) as transaction_count, SUM(CASE WHEN VOLUME > 0 THEN VOLUME ELSE 0 END) as total_in, SUM(CASE WHEN VOLUME < 0 THEN ABS(VOLUME) ELSE 0 END) as total_out, AVG(ABS(VOLUME)) as avg_volume FROM transactions WHERE SKU = ?",
                sku
            )
            
            trans_stats = transactions[0] if transactions else {}
            
            # Get product info
            product_info = get_product_info_safe(sku)
            
            # Calculate turnover
            turnover = trans_stats.get('total_out', 0) / current_stock if current_stock > 0 else 0
            
            analytics_data.append({
                'SKU': sku,
                'Product_Name': product_info.get('NAME', 'Unknown'),
                'Current_Stock': current_stock,
                'Cost': product_info.get('COST', 0),
                'Margin': product_info.get('MARGIN', 0),
                'Transaction_Count': trans_stats.get('transaction_count', 0),
                'Total_Stock_In': trans_stats.get('total_in', 0),
                'Total_Stock_Out': trans_stats.get('total_out', 0),
                'Average_Volume': round(trans_stats.get('avg_volume', 0), 2),
                'Turnover_Ratio': round(turnover, 2),
                'Expiration_Days': product_info.get('EXPIRATION', 365)
            })
        
        df = pd.DataFrame(analytics_data)
        
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=analytics_report.csv"
        response.headers["Content-type"] = "text/csv"
        
        return response
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/import/inventory", methods=["POST"])
def import_inventory():
    """Import inventory data from CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Read CSV
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({"error": f"Error reading CSV: {str(e)}"}), 400
        
        # Validate required columns
        required_columns = ['SKU', 'Current_Stock']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({"error": f"Missing required columns: {missing_columns}"}), 400
        
        # Process import
        successful_imports = 0
        failed_imports = []
        
        for index, row in df.iterrows():
            try:
                sku = str(row['SKU']).strip()
                volume = float(row['Current_Stock'])
                
                if pd.isna(volume) or volume < 0:
                    failed_imports.append(f"Row {index + 1}: Invalid volume")
                    continue
                
                # Use the new transaction processing system
                result = process_transaction(sku, volume, 'Receipts', doc_number=f"IMPORT_{successful_imports + 1}")
                
                if not result.get('success'):
                    failed_imports.append(f"Row {index + 1}: {result.get('error', 'Transaction failed')}")
                    continue
                
                successful_imports += 1
                
            except Exception as e:
                failed_imports.append(f"Row {index + 1}: {str(e)}")
        
        return jsonify({
            "success": True,
            "message": f"Import completed: {successful_imports} successful, {len(failed_imports)} failed",
            "successful_imports": successful_imports,
            "failed_imports": failed_imports
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/import/products", methods=["POST"])
def import_products():
    """Import product information from CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Read CSV
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({"error": f"Error reading CSV: {str(e)}"}), 400
        
        # Validate required columns
        required_columns = ['SKU']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({"error": f"Missing required columns: {missing_columns}"}), 400
        
        # Create or update products table if needed
        execute_query(
            "CREATE TABLE IF NOT EXISTS products (SKU TEXT PRIMARY KEY, NAME TEXT, COST REAL, MARGIN REAL, EXPIRATION INTEGER)"
        )
        
        # Process import
        successful_imports = 0
        failed_imports = []
        
        for index, row in df.iterrows():
            try:
                sku = str(row['SKU']).strip()
                name = str(row.get('Product_Name', 'Unknown')).strip()
                cost = float(row.get('Cost', 0))
                margin = float(row.get('Margin', 0))
                expiration = int(row.get('Expiration_Days', 365))
                
                # Check if product exists
                existing = execute_query("SELECT * FROM products WHERE SKU = ?", sku)
                
                if existing:
                    # Update existing
                    execute_query("""
                        UPDATE products 
                        SET NAME = ?, COST = ?, MARGIN = ?, EXPIRATION = ?
                        WHERE SKU = ?
                    """, name, cost, margin, expiration, sku)
                else:
                    # Insert new
                    execute_query("""
                        INSERT INTO products (SKU, NAME, COST, MARGIN, EXPIRATION)
                        VALUES (?, ?, ?, ?, ?)
                    """, sku, name, cost, margin, expiration)
                
                successful_imports += 1
                
            except Exception as e:
                failed_imports.append(f"Row {index + 1}: {str(e)}")
        
        return jsonify({
            "success": True,
            "message": f"Product import completed: {successful_imports} successful, {len(failed_imports)} failed",
            "successful_imports": successful_imports,
            "failed_imports": failed_imports
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/template/<template_type>")
def download_template(template_type):
    """Download CSV templates for import"""
    try:
        if template_type == 'inventory':
            template_data = [
                {'SKU': 'SAMPLE001', 'Current_Stock': 100},
                {'SKU': 'SAMPLE002', 'Current_Stock': 50},
                {'SKU': 'SAMPLE003', 'Current_Stock': 75}
            ]
        elif template_type == 'products':
            template_data = [
                {'SKU': 'SAMPLE001', 'Product_Name': 'Sample Product 1', 'Cost': 10.00, 'Margin': 25.0, 'Expiration_Days': 365},
                {'SKU': 'SAMPLE002', 'Product_Name': 'Sample Product 2', 'Cost': 15.00, 'Margin': 30.0, 'Expiration_Days': 180},
                {'SKU': 'SAMPLE003', 'Product_Name': 'Sample Product 3', 'Cost': 8.00, 'Margin': 20.0, 'Expiration_Days': 90}
            ]
        else:
            return jsonify({"error": "Invalid template type"}), 400
        
        df = pd.DataFrame(template_data)
        
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = f"attachment; filename={template_type}_template.csv"
        response.headers["Content-type"] = "text/csv"
        
        return response
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

@app.route("/simple_test")
def simple_test():
    """Super simple test"""
    return "Hello World! Flask is working!"

if __name__ == "__main__":
    # For development
    app.run(debug=True, host="0.0.0.0", port=5000)  # Deployment verification - 2025-10-08 23:21:06
