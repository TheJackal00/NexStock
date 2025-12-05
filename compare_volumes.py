import sqlite3

conn = sqlite3.connect('inventory.db')
cursor = conn.cursor()

print("Comparing volume calculations:\n")
print("=" * 80)

# Top SKUs query (sum with sign)
top_skus = cursor.execute(
    "SELECT SKU, SUM(VOLUME) as total_volume, COUNT(*) as transaction_count FROM transactions GROUP BY SKU ORDER BY SKU"
).fetchall()

print("TOP SKUs Chart (SUM with sign - net volume):")
for row in top_skus:
    print(f"  SKU {row[0]}: {row[1]} (from {row[2]} transactions)")

print("\n" + "=" * 80)

# Demand analysis (sum of absolute values)
print("\nDemand Analysis (SUM of ABS - total movement):")
for sku_data in top_skus:
    sku = sku_data[0]
    volumes = cursor.execute("SELECT VOLUME FROM transactions WHERE SKU = ?", (sku,)).fetchall()
    vol_values = [abs(v[0]) for v in volumes]
    total_volume = sum(vol_values)
    print(f"  SKU {sku}: {total_volume} (from {len(volumes)} transactions)")

print("\n" + "=" * 80)
print("\nBreakdown for SKU 001:")
volumes = cursor.execute("SELECT VOLUME FROM transactions WHERE SKU = '001'").fetchall()
print(f"Individual volumes: {[v[0] for v in volumes]}")
print(f"Sum with sign: {sum([v[0] for v in volumes])}")
print(f"Sum of absolute values: {sum([abs(v[0]) for v in volumes])}")

conn.close()
