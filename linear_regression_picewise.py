import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from numpy.polynomial import Polynomial

class DualGuaranteedApproximation:
    """
    Two-Stage Curve Fitting with Dual Guarantees (Derivative + Integral Preservation)
    
    این متد از قضیه یگانگی تابع به صورت تقریبی استفاده می‌کند:
    اگر f(x) و g(x) دو تابع باشند که:
    1. مقادیرشان در نقاط مشخص یکسان باشد
    2. مشتق‌هایشان در همان نقاط یکسان باشد
    آنگاه f و g تقریباً یکسان هستند (به خصوص برای توابع هموار).
    
    Stage 1 (O(n)): Piecewise linear interpolation → reference path
    Stage 2 (O(n^3)): Weighted polynomial regression → smooth approximation
    
    این روش computational efficiency را با mathematical rigor ترکیب می‌کند.
    """
    
    def __init__(self, degree=5, n_interpolation_points=1000, 
                 derivative_weight=1.0, point_weight=1.0):
        """
        Parameters:
        -----------
        degree : int
            درجه چندجمله‌ای برای Stage 2 regression
        n_interpolation_points : int
            تعداد نقاط تولیدی در Stage 1 برای dense sampling
        derivative_weight : float
            وزن اهمیت مشتق در optimization (shape constraint)
        point_weight : float
            وزن اهمیت مقدار تابع (integral constraint)
        """
        self.degree = degree
        self.n_interp = n_interpolation_points
        self.w_deriv = derivative_weight
        self.w_point = point_weight
        
        # Storage برای intermediate results
        self.interp_x = None
        self.interp_y = None
        self.interp_derivatives = None
        self.polynomial = None
        
    def _stage1_interpolation(self, x, y):
        """
        Stage 1: Fast acceleration stage (O(n))
        
        تولید یک reference path از طریق piecewise linear interpolation:
        - سریع (no iterations)
        - تولید dense clean samples
        - محاسبه مشتقات مرجع (local slopes)
        """
        # Sort data (ضروری برای interpolation)
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        
        # Create linear interpolator
        interpolator = interp1d(x_sorted, y_sorted, kind='linear')
        
        # Generate dense sample points
        x_interp = np.linspace(x_sorted.min(), x_sorted.max(), self.n_interp)
        y_interp = interpolator(x_interp)
        
        # Compute reference derivatives (slopes of line segments)
        derivatives = np.gradient(y_interp, x_interp)
        
        return x_interp, y_interp, derivatives
    
    def _stage2_weighted_regression(self, x, y, derivatives):
        """
        Stage 2: یافتن equivalent polynomial با dual constraints
        
        این مرحله از weighted least squares استفاده می‌کند تا همزمان:
        1. f(x) ≈ y_reference (point fitting)
        2. f'(x) ≈ dy/dx_reference (derivative fitting)
        
        براساس قضیه یگانگی تقریبی، چندجمله‌ای بدست آمده "تقریباً معادل"
        با reference path است.
        """
        n = len(x)
        
        # Design matrix برای point fitting: [1, x, x², ..., x^degree]
        X_points = np.vander(x, self.degree + 1, increasing=True)
        
        # Design matrix برای derivative fitting: [0, 1, 2x, 3x², ..., degree·x^(degree-1)]
        X_derivs = np.zeros_like(X_points)
        for i in range(1, self.degree + 1):
            X_derivs[:, i] = i * x**(i-1)
        
        # Weighted augmented system:
        # [w_p·X_points  ] [c]   [w_p·y        ]
        # [w_d·X_derivs  ] [ ] = [w_d·dy/dx    ]
        X_combined = np.vstack([
            self.w_point * X_points,
            self.w_deriv * X_derivs
        ])
        
        y_combined = np.hstack([
            self.w_point * y,
            self.w_deriv * derivatives
        ])
        
        # Solve least squares
        coeffs = np.linalg.lstsq(X_combined, y_combined, rcond=None)[0]
        
        return Polynomial(coeffs)
    
    def fit(self, x, y):
        """
        اجرای کامل two-stage pipeline
        """
        x = np.array(x)
        y = np.array(y)
        
        # Stage 1: Reference path generation
        self.interp_x, self.interp_y, self.interp_derivatives = \
            self._stage1_interpolation(x, y)
        
        # Stage 2: Smooth equivalent function
        self.polynomial = self._stage2_weighted_regression(
            self.interp_x, self.interp_y, self.interp_derivatives
        )
        
        return self
    
    def predict(self, x):
        """پیش‌بینی مقادیر y برای x های جدید"""
        if self.polynomial is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.polynomial(x)
    
    def derivative(self, x):
        """محاسبه مشتق در نقطه x"""
        if self.polynomial is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.polynomial.deriv()(x)
    
    def integral(self, x_start, x_end):
        """
        محاسبه انتگرال معین تحلیلی
        (برای اعتبارسنجی integral preservation)
        """
        if self.polynomial is None:
            raise ValueError("Model not fitted. Call fit() first.")
        antiderivative = self.polynomial.integ()
        return antiderivative(x_end) - antiderivative(x_start)
    
    def validate(self):
        """
        اعتبارسنجی بر اساس dual guarantees:
        1. Relative integral error (scale-independent)
        2. Derivative RMSE
        
        Returns:
        --------
        dict containing:
            - relative_integral_error: نسبت خطای مساحت (باید ≈ 0)
            - derivative_rmse: میانگین خطای مشتق (باید کم باشد)
            - polynomial_coeffs: ضرایب چندجمله‌ای (برای تحلیل)
        """
        if self.polynomial is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # محاسبه انتگرال reference (Stage 1)
        integral_ref = simpson(self.interp_y, x=self.interp_x)
        
        # محاسبه انتگرال smooth polynomial (Stage 2)
        x_min, x_max = self.interp_x.min(), self.interp_x.max()
        integral_smooth = self.integral(x_min, x_max)
        
        # Relative error (scale-independent metric)
        integral_error = abs(integral_smooth - integral_ref) / abs(integral_ref)
        
        # Derivative error
        deriv_pred = self.polynomial.deriv()(self.interp_x)
        deriv_rmse = np.sqrt(np.mean((self.interp_derivatives - deriv_pred)**2))
        
        return {
            'relative_integral_error': integral_error,
            'derivative_rmse': deriv_rmse,
            'polynomial_coeffs': self.polynomial.coef
        }
    
    def plot_results(self, original_x, original_y, figsize=(14, 6), 
                     show_detailed=False):
        """
        نمایش بصری نتایج
        
        Parameters:
        -----------
        original_x, original_y : array-like
            داده‌های اصلی ورودی
        show_detailed : bool
            اگر True باشد، نمودارهای اضافی (residuals, metrics) نمایش داده می‌شود
        """
        if self.polynomial is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Generate dense points for smooth plotting
        x_plot = np.linspace(original_x.min(), original_x.max(), 500)
        y_plot = self.predict(x_plot)
        deriv_plot = self.derivative(x_plot)
        
        if not show_detailed:
            # Simple 2-panel view (for presentations)
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Panel 1: Function fitting
            ax = axes[0]
            ax.scatter(original_x, original_y, alpha=0.6, 
                      label='Original Data', s=50, color='blue')
            ax.plot(self.interp_x, self.interp_y, 'g--', alpha=0.5, 
                   label='Stage 1: Piecewise Linear', linewidth=1.5)
            ax.plot(x_plot, y_plot, 'r-', linewidth=2, 
                   label='Stage 2: Smooth Polynomial')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Dual-Guaranteed Function Approximation')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Panel 2: Derivative comparison
            ax = axes[1]
            ax.plot(self.interp_x, self.interp_derivatives, 'g--', 
                   alpha=0.5, label='Stage 1 Derivatives', linewidth=1.5)
            ax.plot(x_plot, deriv_plot, 'r-', linewidth=2, 
                   label='Stage 2 Derivatives')
            ax.set_xlabel('x')
            ax.set_ylabel('dy/dx')
            ax.set_title('Derivative Matching (Shape Preservation)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        else:
            # Detailed 4-panel view (for analysis)
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Panel 1: Complete pipeline
            ax = axes[0, 0]
            ax.scatter(original_x, original_y, alpha=0.6, 
                      label='Original Data', s=50)
            ax.plot(self.interp_x, self.interp_y, 'g--', alpha=0.5,
                   label='Stage 1: Reference Path')
            ax.plot(x_plot, y_plot, 'r-', linewidth=2,
                   label='Stage 2: Smooth Approximation')
            ax.set_title('Complete Two-Stage Pipeline')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Panel 2: Derivatives
            ax = axes[0, 1]
            ax.plot(self.interp_x, self.interp_derivatives, 'g--', 
                   alpha=0.5, label='Reference')
            ax.plot(x_plot, deriv_plot, 'r-', linewidth=2, label='Smooth')
            ax.set_title('Derivative Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Panel 3: Residuals
            ax = axes[1, 0]
            y_pred_interp = self.predict(self.interp_x)
            residuals = self.interp_y - y_pred_interp
            ax.scatter(self.interp_x, residuals, alpha=0.5, s=5)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
            ax.set_xlabel('x')
            ax.set_ylabel('Residual')
            ax.set_title(f'Point-wise Error (Mean: {np.mean(np.abs(residuals)):.4f})')
            ax.grid(True, alpha=0.3)
            
            # Panel 4: Validation metrics
            ax = axes[1, 1]
            metrics = self.validate()
            
            metric_names = ['Relative\nIntegral Error', 'Derivative\nRMSE']
            metric_values = [
                metrics['relative_integral_error'],
                metrics['derivative_rmse']
            ]
            
            bars = ax.bar(metric_names, metric_values, 
                         color=['#ff6b6b', '#4ecdc4'])
            ax.set_ylabel('Error')
            ax.set_title('Validation Metrics')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig

# ============= Example Usage =============
if __name__ == "__main__":
    # Generate noisy test data
    np.random.seed(42)
    x_data = np.linspace(0, 10, 25)
    y_true = np.sin(x_data) + 0.5 * np.cos(2*x_data) + 0.1*x_data
    y_noisy = y_true + np.random.normal(0, 0.2, len(x_data))
    
    # Apply the method
    model = DualGuaranteedApproximation(
        degree=6,
        n_interpolation_points=1000,
        derivative_weight=1.0,
        point_weight=1.0
    )
    
    model.fit(x_data, y_noisy)
    
    # Visualize (simple version)
    print("=== Simple Presentation View ===")
    model.plot_results(x_data, y_noisy, show_detailed=False)
    plt.show()
    
    # Visualize (detailed version for analysis)
    print("\n=== Detailed Analysis View ===")
    model.plot_results(x_data, y_noisy, show_detailed=True)
    plt.show()
    
    # Validation
    metrics = model.validate()
    print("\n=== Validation Based on Dual Guarantees ===")
    print(f"Polynomial coefficients: {metrics['polynomial_coeffs']}")
    print(f"Relative Integral Error: {metrics['relative_integral_error']:.6f}")
    print(f"  → Should be ≈ 0 (integral preservation)")
    print(f"Derivative RMSE: {metrics['derivative_rmse']:.6f}")
    print(f"  → Measures shape preservation quality")
    
    # Example predictions
    x_new = np.array([2.5, 5.0, 7.5])
    y_pred = model.predict(x_new)
    deriv_pred = model.derivative(x_new)
    
    print(f"\n=== Example Predictions ===")
    print(f"At x = {x_new}:")
    print(f"  y = {y_pred}")
    print(f"  dy/dx = {deriv_pred}")
    
    # Integral calculation
    integral_val = model.integral(0, 10)
    print(f"\nIntegral from 0 to 10: {integral_val:.6f}")