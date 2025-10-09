#!/usr/bin/env python3
"""
Script para ejecutar consultas SQL en la base de datos NexStock
Uso: python sql_query.py
"""

import sqlite3
import os
import tabulate

def get_db_connection():
    """Get a database connection"""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inventory.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def execute_query(query):
    """Execute a query and return results"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        
        if query.strip().upper().startswith('SELECT'):
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        else:
            conn.commit()
            return f"Query executed successfully. Rows affected: {cursor.rowcount}"
    except Exception as e:
        return f"Error: {e}"
    finally:
        conn.close()

def show_tables():
    """Show all tables in the database"""
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    results = execute_query(query)
    if isinstance(results, list):
        print("\n📋 Tablas disponibles:")
        for table in results:
            print(f"  - {table['name']}")
    return results

def describe_table(table_name):
    """Show table structure"""
    query = f"PRAGMA table_info({table_name});"
    results = execute_query(query)
    if isinstance(results, list):
        print(f"\n📊 Estructura de la tabla '{table_name}':")
        headers = ["Column", "Type", "NotNull", "Default", "PrimaryKey"]
        rows = []
        for col in results:
            rows.append([
                col['name'], 
                col['type'], 
                "YES" if col['notnull'] else "NO",
                col['dflt_value'] or "NULL",
                "YES" if col['pk'] else "NO"
            ])
        print(tabulate.tabulate(rows, headers=headers, tablefmt="grid"))
    return results

def interactive_mode():
    """Interactive SQL query mode"""
    print("🚀 NexStock SQL Query Tool")
    print("=" * 50)
    
    # Show available tables
    show_tables()
    
    print("\n💡 Comandos especiales:")
    print("  .tables    - Mostrar todas las tablas")
    print("  .desc <tabla> - Describir estructura de tabla")
    print("  .quit      - Salir")
    print("\n✨ Ejemplos de consultas:")
    print("  SELECT * FROM products LIMIT 5;")
    print("  SELECT SKU, NAME, MARGIN FROM products;")
    print("  SELECT COUNT(*) FROM inventory;")
    
    while True:
        try:
            query = input("\n🔍 SQL> ").strip()
            
            if not query:
                continue
                
            if query.lower() == '.quit':
                print("👋 ¡Hasta luego!")
                break
            elif query.lower() == '.tables':
                show_tables()
                continue
            elif query.lower().startswith('.desc '):
                table_name = query[6:].strip()
                describe_table(table_name)
                continue
            
            if not query.endswith(';'):
                query += ';'
            
            print(f"\n⚡ Ejecutando: {query}")
            results = execute_query(query)
            
            if isinstance(results, list):
                if results:
                    # Format results as table
                    headers = list(results[0].keys())
                    rows = [[row[col] for col in headers] for row in results]
                    print(f"\n📊 Resultados ({len(results)} filas):")
                    print(tabulate.tabulate(rows, headers=headers, tablefmt="grid"))
                else:
                    print("📭 Sin resultados")
            else:
                print(f"✅ {results}")
                
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    try:
        # Install tabulate if not available
        import tabulate
    except ImportError:
        print("📦 Instalando dependencia tabulate...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
        import tabulate
    
    interactive_mode()