# Smart Engine Health Monitoring: Machine Learning Framework for Turbofan RUL Prediction
**Machine Learning Framework for Aircraft Turbofan Engine RUL Prediction**

This research project developed a robust machine learning framework to predict the Remaining Useful Life (RUL) of aircraft turbofan engines, enabling proactive maintenance strategies and reducing operational risks in aviation.

## Objective
To create a comprehensive predictive maintenance solution that:
- **Predicts Remaining Useful Life (RUL)** of turbofan engines with high accuracy
- **Evaluates multiple machine learning models** including regression and classification techniques
- **Analyzes critical sensor data** to identify key degradation indicators
- **Enables proactive maintenance scheduling** to minimize unplanned failures

## Dataset
**NASA C-MAPSS Dataset** (Commercial Modular Aero-Propulsion System Simulation)
- **Source**: NASA Prognostics Center of Excellence
- **Size**: Multivariate time-series data from 100 engines across their complete lifecycles
- **Features**: 21 sensor measurements + 3 operational settings per engine
- **Structure**: Training data includes full lifecycle to failure; test data ends before failure for RUL prediction
- **Challenges**: Imbalanced dataset with high dimensionality requiring extensive feature engineering

## Methodology

### Machine Learning Models Evaluated:
**Regression Models** (for RUL prediction):
- Linear Regression (baseline)
- Support Vector Regression (SVR)
- Random Forest Regression
- XGBoost Regression

**Classification Models** (for engine health categorization):
- Support Vector Machines (SVM)
- Naïve Bayes
- K-Nearest Neighbors (KNN)
- Random Forest Classifier

### Feature Engineering:
- **Correlation analysis** to identify key predictive features
- **Rolling mean features** (10-cycle windows) to capture temporal degradation patterns
- **Feature selection** removing constant/irrelevant sensors
- **MinMax scaling** for sensor data normalization
- **RUL clipping** at 195 cycles to reduce outliers and improve accuracy

### Risk Categorization:
- **Risk Zone** (RUL ≤ 68): Immediate attention required
- **Moderate Risk Zone** (69 ≤ RUL ≤ 137): Maintenance scheduling advised  
- **No Risk Zone** (RUL > 137): Normal operations

## Key Results

### Best Performing Models:
| Model Type | Best Algorithm | Performance Metric | Score |
|------------|---------------|-------------------|-------|
| **Regression** | Support Vector Regression (SVR) | Test RMSE | **26.59** |
| **Classification** | Support Vector Machine (SVM) | Test Accuracy | **68.3%** |

### Model Performance Comparison:
| Model | Test RMSE | R² Score | Performance Notes |
|-------|-----------|----------|-------------------|
| **SVR** | **26.59** | **0.48** | Most consistent, best generalization |
| Random Forest | 29.95 | 0.75 | High accuracy but prone to overfitting |
| XGBoost | 30.45 | 0.74 | Good performance, handles imbalanced data well |
| Linear Regression | 55.62 | 0.32 | Limited by linear assumptions |

### Critical Features Identified:
1. **Physical Core Speed (s_9)** - Primary degradation indicator
2. **HPC Outlet Static Pressure (s_11)** - Critical pressure monitoring
3. **Fuel Flow to Ps30 Ratio (s_12)** - Efficiency indicator

### Classification Performance:
- **Risk Zone Detection**: 88% recall (excellent at identifying critical engines)
- **Moderate Risk Zone**: 56% recall (room for improvement)
- **No Risk Zone**: 67% recall (moderate performance)

## 🚀 Business Impact

### Operational Benefits:
- **Proactive Maintenance**: Transition from reactive to predictive maintenance strategies
- **Cost Reduction**: Minimize unplanned downtime and avoid expensive emergency repairs
- **Safety Enhancement**: Early identification of high-risk engines prevents catastrophic failures
- **Resource Optimization**: Improved maintenance scheduling and resource allocation

### Financial Implications:
- **Reduced Maintenance Costs**: Elimination of unnecessary preventive maintenance
- **Minimized Downtime**: Optimized scheduling reduces operational interruptions
- **Extended Asset Lifespan**: Better maintenance timing maximizes engine lifecycle value

## Technical Stack

**Programming Language**: Python 3.x

**Key Libraries**:
- `pandas`, `numpy` - Data manipulation and analysis
- `scikit-learn` - Machine learning models and evaluation
- `xgboost` - Gradient boosting framework
- `matplotlib`, `seaborn` - Data visualization


## 🔍 Key Insights

### Model Optimization Journey:
1. **Attempt 1**: Baseline models on full dataset - identified overfitting issues
2. **Attempt 2**: Removed irrelevant sensors, applied RUL clipping - improved stability  
3. **Attempt 3**: Added historical features (rolling means) - **significant performance boost**

### Feature Engineering Success:
- Dropping 7 constant/irrelevant sensors improved model focus
- Rolling mean features captured crucial temporal degradation patterns
- RUL clipping reduced prediction variance and outlier impact

### Risk Zone Analysis:
- **High accuracy** in identifying critical engines (Risk Zone: 88% recall)
- **Moderate success** in borderline cases (Moderate Risk Zone: 56% recall)
- Provides actionable maintenance prioritization framework



## Contributors
- **Sameer Shaik** 
- **Teja Swaroop Kotharu**  
- **Thejesh Reddy Marripati** 
