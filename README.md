# NexStock - Inventory Management System

A comprehensive inventory management system with Monte Carlo simulation and linear programming optimization for purchase scheduling.

## Features

- **Portfolio Management**: Track products with SKUs, names, margins, and costs
- **Inventory Tracking**: Monitor stock levels, expiration dates, and volumes
- **Transaction History**: Record and analyze all inventory movements
- **Monte Carlo Simulation**: Generate demand forecasts with configurable lead times
- **Optimization Engine**: Use Integer Linear Programming to optimize purchase schedules
- **Results Dashboard**: Visualize optimized purchasing calendars and strategies

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Database**: SQLite with custom query functions
- **Optimization**: PuLP (Python Linear Programming)
- **Simulation**: NumPy for Monte Carlo analysis
- **Frontend**: HTML/CSS with Bootstrap styling

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Access the application at `http://localhost:5000`

## Deployment

This application is configured for deployment on Render.com with the following files:
- `requirements.txt`: Python dependencies
- `Procfile`: Process configuration for web server
- `build.sh`: Build script for deployment
- `runtime.txt`: Python version specification

## API Endpoints

- `/` - Home page
- `/portafolio` - Product portfolio management
- `/inventory` - Current inventory status
- `/transactions` - Transaction history
- `/simulate` - Run Monte Carlo simulations
- `/optimize` - Generate optimal purchase schedules
- `/results` - View optimization results

## Database Schema

The application uses SQLite with the following main tables:
- `products` - Product information (SKU, name, margin, cost)
- `inventory` - Current stock levels and expiration data
- `transactions` - Historical transaction records
- `Simulation` - Monte Carlo simulation results

## Author

Developed for inventory optimization and demand forecasting analysis.