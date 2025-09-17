


// Enhanced tariff_v4.js with complete FCAS integration and Load Data Manager
let formConfig = [];
let addressSectionInitialized = false;
let tariffSectionInitialized = false;
let fcasSectionInitialized = false;
let loadSectionInitialized = false;

console.log("🚀 Starting complete enhanced form with FCAS and Load integration...");

// Enhanced CSS with FCAS and Load-specific styles
function injectEnhancedCSS() {
    const cssContent = `
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem 0;
            margin-bottom: 2rem;
        }
        .form-card {
            border: none;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            margin-bottom: 1.5rem;
        }
        .form-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        .nav-tabs-custom {
            border-bottom: 2px solid #e2e8f0;
            margin-bottom: 2rem;
        }
        .nav-tabs-custom .nav-link {
            border: none;
            color: #718096;
            font-weight: 500;
            padding: 1rem 1.5rem;
            border-radius: 10px 10px 0 0;
            margin-right: 0.5rem;
            transition: all 0.3s ease;
        }
        .nav-tabs-custom .nav-link:hover {
            background-color: #f8f9fa;
            color: #667eea;
        }
        .nav-tabs-custom .nav-link.active {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            color: white !important;
            border: none;
        }
        .form-control {
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            transition: all 0.3s ease;
        }
        .form-control:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            outline: none;
        }
        .progress-indicator {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transform: scaleX(0);
            transform-origin: left;
            transition: transform 0.3s ease;
            z-index: 1000;
        }
        .address-section {
            background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1.5rem 0;
            border: 1px solid #bae6fd;
        }
        .subsection-header {
            background: linear-gradient(135deg, #f7fafc, #edf2f7);
            border-left: 4px solid #667eea;
            padding: 1rem 1.5rem;
            margin: 1.5rem 0 1rem 0;
            border-radius: 0 8px 8px 0;
        }
        .subsection-header h4 {
            margin: 0;
            color: #2d3748;
            font-size: 1.1rem;
            font-weight: 600;
        }
        #map {
            height: 300px;
            width: 100%;
            border-radius: 8px;
            border: 2px solid #e2e8f0;
            margin-top: 1rem;
            background-color: #f0f0f0;
        }

        /* Tariff-specific styles */
        .tariff-section {
            background: linear-gradient(135deg, #fff5e6, #fff0db);
            border: 1px solid #ffd6a5;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1.5rem 0;
        }
        .tou-schedule-grid {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .tou-period-row {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 0.75rem;
            margin: 0.5rem 0;
        }
        .demand-charge-section {
            background: linear-gradient(135deg, #fef5e7, #fdf2e9);
            border: 1px solid #fed7aa;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .rate-input-group {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }

        /* FCAS-specific styles */
        .fcas-section {
            background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
            border: 1px solid #7dd3fc;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1.5rem 0;
        }
        .fcas-service-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        .fcas-service-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
        }
        .fcas-service-card.enabled {
            border-color: #10b981;
            background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        }
        .fcas-auto-config {
            background: linear-gradient(135deg, #fffbeb, #fef3c7);
            border: 1px solid #fbbf24;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .fcas-revenue-estimate {
            background: linear-gradient(135deg, #f0fdf4, #dcfce7);
            border: 1px solid #10b981;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            text-align: center;
        }

        /* Load Data specific styles */
        .load-section {
            background: linear-gradient(135deg, #fef3c7, #fde68a);
            border: 1px solid #f59e0b;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1.5rem 0;
        }
        .load-method-card {
            background: white;
            border: 2px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .load-method-card:hover {
            border-color: #3b82f6;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
        }
        .load-method-card.selected {
            border-color: #10b981;
            background: linear-gradient(135deg, #ecfdf5, #d1fae5);
        }
        .load-method-card .method-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        .upload-area {
            border: 2px dashed #d1d5db;
            border-radius: 8px;
            padding: 2rem;
            text-align: center;
            background: #f9fafb;
            transition: all 0.3s ease;
        }
        .upload-area.drag-over {
            border-color: #3b82f6;
            background: #eff6ff;
        }
        .upload-progress {
            background: #f3f4f6;
            border-radius: 4px;
            height: 8px;
            overflow: hidden;
            margin: 1rem 0;
        }
        .upload-progress-bar {
            background: linear-gradient(90deg, #10b981, #059669);
            height: 100%;
            transition: width 0.3s ease;
        }
        .monthly-input-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        .month-input-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 1rem;
        }
        .load-validation-result {
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .load-validation-result.success {
            background: #ecfdf5;
            border: 1px solid #10b981;
            color: #065f46;
        }
        .load-validation-result.error {
            background: #fef2f2;
            border: 1px solid #ef4444;
            color: #991b1b;
        }
        .load-chart-container {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            height: 300px;
        }
        .spinner {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
    `;

    const styleElement = document.createElement('style');
    styleElement.textContent = cssContent;
    document.head.appendChild(styleElement);
}

// Tab configuration with Load
const TAB_CONFIG = {
    'Project Information': { icon: 'fas fa-building', order: 1 },
    'PV Configuration': { icon: 'fas fa-solar-panel', order: 2 },
    'Battery Storage': { icon: 'fas fa-battery-three-quarters', order: 3 },
    'Network Configuration': { icon: 'fas fa-plug', order: 4 },
    'Financial Parameters': { icon: 'fas fa-dollar-sign', order: 5 },
    'Project Costs': { icon: 'fas fa-calculator', order: 6 },
    'Load': { icon: 'fas fa-chart-line', order: 7 }
};

// ========== LOAD DATA MANAGER IMPLEMENTATION ==========

class LoadDataManager {
    constructor() {
        this.currentMethod = 'auto';
        this.loadData = null;
        this.validationResult = null;
        this.chart = null;
        this.monthlyProfiles = {
            residential: [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.8],
            commercial: [1.1, 1.0, 0.9, 0.9, 1.0, 1.1, 1.2, 1.2, 1.1, 1.0, 1.0, 1.1],
            industrial: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        };
        this.hourlyProfiles = {
            residential: [0.4, 0.3, 0.3, 0.3, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.5, 0.5,
                         0.6, 0.7, 0.8, 0.9, 1.2, 1.5, 1.8, 1.6, 1.2, 0.9, 0.7, 0.5],
            commercial: [0.2, 0.2, 0.2, 0.2, 0.3, 0.5, 0.8, 1.2, 1.5, 1.6, 1.7, 1.8,
                        1.8, 1.7, 1.6, 1.5, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2],
            industrial: [0.9, 0.8, 0.8, 0.8, 0.9, 1.0, 1.1, 1.2, 1.2, 1.2, 1.2, 1.2,
                        1.2, 1.2, 1.2, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9]
        };

        this.initialized = false;
    }

    initialize() {
        if (this.initialized) return;
        this.initialized = true;
        console.log("🔌 Load Data Manager initialized");
    }

    getCurrentMethod() {
        return this.currentMethod;
    }

    getSummary() {
        switch (this.currentMethod) {
            case 'file':
                return this.loadData ? `Custom data loaded (${this.loadData.length} hours)` : 'No file uploaded';
            case 'manual':
                return 'Custom monthly data configured';
            case 'auto':
            default:
                const annualEnergy = document.getElementById('load-annual-energy')?.value || '10000';
                const loadType = document.getElementById('load-type-select')?.value || 'residential';
                return `Auto-generated: ${annualEnergy} kWh/year (${loadType})`;
        }
    }

    hasChart() {
        return this.chart !== null;
    }

    getFormData() {
        const data = {
            load_method: this.currentMethod,
            load_validation_passed: this.validationResult?.valid || false
        };

        switch (this.currentMethod) {
            case 'file':
                if (this.loadData) {
                    data.load_data = this.loadData;
                    data.load_data_source = 'uploaded_file';
                }
                break;
            case 'manual':
                const monthlyData = [];
                for (let i = 1; i <= 12; i++) {
                    const input = document.getElementById(`month-${i}-energy`);
                    monthlyData.push(parseFloat(input?.value) || 0);
                }
                data.load_monthly_data = monthlyData;
                data.load_data_source = 'manual_monthly';

                const loadType = document.getElementById('load-type-select')?.value || 'residential';
                data.load_type = loadType;
                break;
            case 'auto':
            default:
                const annualEnergy = parseFloat(document.getElementById('load-annual-energy')?.value) || 10000;
                const autoLoadType = document.getElementById('load-type-select')?.value || 'residential';
                data.load_annual_energy = annualEnergy;
                data.load_type = autoLoadType;
                data.load_data_source = 'auto_generated';
                break;
        }

        return data;
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('load_file', file);

        try {
            const response = await fetch('/upload-load-data', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.status}`);
            }

            const result = await response.json();

            if (result.success) {
                this.loadData = result.data;
                this.validationResult = result.validation;
                this.currentMethod = 'file';
                return { success: true, data: result };
            } else {
                throw new Error(result.error || 'Upload failed');
            }
        } catch (error) {
            console.error('Load file upload error:', error);
            return { success: false, error: error.message };
        }
    }

    async validateLoadData(data, method) {
        try {
            const response = await fetch('/validate-load-data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ data, method })
            });

            if (!response.ok) {
                throw new Error(`Validation failed: ${response.status}`);
            }

            const result = await response.json();
            this.validationResult = result;
            return result;
        } catch (error) {
            console.error('Load validation error:', error);
            return { valid: false, error: error.message };
        }
    }

    generateHourlyProfile(annualEnergy, loadType = 'residential') {
        const monthlyFactors = this.monthlyProfiles[loadType] || this.monthlyProfiles.residential;
        const hourlyFactors = this.hourlyProfiles[loadType] || this.hourlyProfiles.residential;

        const profile = [];
        const daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

        for (let month = 0; month < 12; month++) {
            const monthlyFactor = monthlyFactors[month];
            const daysInThisMonth = daysInMonth[month];

            for (let day = 0; day < daysInThisMonth; day++) {
                for (let hour = 0; hour < 24; hour++) {
                    const hourlyFactor = hourlyFactors[hour];
                    const baseLoad = annualEnergy / 8760;
                    const adjustedLoad = baseLoad * monthlyFactor * hourlyFactor;
                    profile.push(adjustedLoad);
                }
            }
        }

        return profile;
    }

    generateFromMonthlyData(monthlyData, loadType = 'residential') {
        const hourlyFactors = this.hourlyProfiles[loadType] || this.hourlyProfiles.residential;
        const profile = [];
        const daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

        for (let month = 0; month < 12; month++) {
            const monthlyEnergy = monthlyData[month] || 0;
            const daysInThisMonth = daysInMonth[month];
            const hoursInMonth = daysInThisMonth * 24;
            const baseHourlyLoad = monthlyEnergy / hoursInMonth;

            for (let day = 0; day < daysInThisMonth; day++) {
                for (let hour = 0; hour < 24; hour++) {
                    const hourlyFactor = hourlyFactors[hour];
                    const adjustedLoad = baseHourlyLoad * hourlyFactor;
                    profile.push(adjustedLoad);
                }
            }
        }

        return profile;
    }

    createChart(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) return null;

        container.innerHTML = '<canvas id="load-chart-canvas"></canvas>';
        const canvas = document.getElementById('load-chart-canvas');

        if (!canvas || typeof Chart === 'undefined') {
            container.innerHTML = '<div class="text-center text-muted p-4">Chart library not available</div>';
            return null;
        }

        const ctx = canvas.getContext('2d');

        const weekData = data.slice(0, 168);
        const labels = [];
        for (let day = 0; day < 7; day++) {
            for (let hour = 0; hour < 24; hour++) {
                labels.push(`Day ${day + 1} ${hour}:00`);
            }
        }

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Load (kWh)',
                    data: weekData,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { display: false },
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Load (kWh)' }
                    }
                },
                plugins: {
                    legend: { display: true },
                    title: { display: true, text: 'Load Profile (First Week)' }
                }
            }
        });

        return this.chart;
    }

    destroyChart() {
        if (this.chart) {
            this.chart.destroy();
            this.chart = null;
        }
    }
}

// Initialize global load manager
window.loadDataManager = new LoadDataManager();

// ========== LOAD SECTION CREATION ==========

function createLoadSection() {
    // Get default values from form config
    const gridFields = formConfig.filter(f => f.section === "Load");
    const getDefault = (key) => {
        const field = gridFields.find(f => f.key === key);
        return field?.default_value || '';
    };

    return `
        <!-- Grid Connection Configuration Section -->
        <div class="subsection-header">
            <h4><i class="fas fa-plug me-2"></i>Grid Connection Configuration</h4>
        </div>
        <div class="rate-input-group mb-4">
            <div class="row g-3">
                <div class="col-md-6">
                    <label for="enable_interconnection_limit" class="form-label fw-semibold">
                        Enable Grid Interconnection Limit?
                    </label>
                    <select class="form-control"
                            id="enable_interconnection_limit"
                            name="enable_interconnection_limit"
                            data-pysam-key="grid.enable_interconnection_limit">
                        <option value="yes" ${getDefault('enable_interconnection_limit') === 'yes' ? 'selected' : ''}>Yes</option>
                        <option value="no" ${getDefault('enable_interconnection_limit') === 'no' ? 'selected' : ''}>No</option>
                    </select>
                    <div class="form-text">Controls whether to enforce AC power limits to/from grid</div>
                </div>
                <div class="col-md-6">
                    <label for="grid_interconnection_limit_kwac" class="form-label fw-semibold">
                        Grid Interconnection Limit (kW AC)
                    </label>
                    <input type="number"
                           class="form-control"
                           id="grid_interconnection_limit_kwac"
                           name="grid_interconnection_limit_kwac"
                           data-pysam-key="grid.grid_interconnection_limit_kwac"
                           value="${getDefault('grid_interconnection_limit_kwac')}"
                           min="0"
                           step="0.1"
                           placeholder="Optional">
                    <div class="form-text">Maximum AC power allowed to/from grid</div>
                </div>
            </div>
            <div class="row g-3 mt-2">
                <div class="col-md-4">
                    <label for="load_escalation" class="form-label fw-semibold">
                        Load Escalation (%)
                        <span class="text-danger">*</span>
                    </label>
                    <input type="number"
                           class="form-control"
                           id="load_escalation"
                           name="load_escalation"
                           data-pysam-key="grid.load_escalation"
                           value="${getDefault('load_escalation')}"
                           min="-10"
                           max="20"
                           step="0.1"
                           required>
                    <div class="form-text">Annual load growth rate (%/year)</div>
                </div>
                <div class="col-md-4">
                    <label for="load_step" class="form-label fw-semibold">
                        Data Resolution
                        <span class="text-danger">*</span>
                    </label>
                    <select class="form-control"
                            id="load_step"
                            name="load_step"
                            data-pysam-key="grid.load_step"
                            required>
                        <option value="1" ${getDefault('load_step') === '1' ? 'selected' : ''}>Hourly (8760 points)</option>
                        <option value="24" ${getDefault('load_step') === '24' ? 'selected' : ''}>Daily (365 points)</option>
                    </select>
                    <div class="form-text">Time resolution for load data analysis</div>
                </div>
                <div class="col-md-4">
                    <label for="load_peak" class="form-label fw-semibold">
                        Peak Demand (kW)
                    </label>
                    <input type="number"
                           class="form-control"
                           id="load_peak"
                           name="load_peak"
                           data-pysam-key="grid.load_peak"
                           value="${getDefault('load_peak')}"
                           min="0"
                           step="0.1"
                           placeholder="Auto-calculated from profile">
                    <div class="form-text">Maximum load in profile (auto-calculated if empty)</div>
                </div>
            </div>
        </div>

        <!-- Annual Energy Fallback -->
        <div class="subsection-header">
            <h4><i class="fas fa-battery-three-quarters me-2"></i>Load Data Configuration</h4>
        </div>
        <div class="rate-input-group mb-4">
            <div class="row g-3">
                <div class="col-md-6">
                    <label for="annual_energy" class="form-label fw-semibold">
                        Annual AC Energy in Year 1 (kWh)
                    </label>
                    <input type="number"
                           class="form-control"
                           id="annual_energy"
                           name="annual_energy"
                           data-pysam-key="grid.annual_energy"
                           value="${getDefault('annual_energy')}"
                           min="0"
                           step="1">
                    <div class="form-text">Used only if no detailed load profile is provided</div>
                </div>
                <div class="col-md-6">
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle me-2"></i>
                        <strong>Load Data Priority:</strong><br>
                        1. Custom CSV upload (preferred)<br>
                        2. Auto-generated profile<br>
                        3. Annual energy fallback
                    </div>
                </div>
            </div>
        </div>

        <div class="load-section" id="load-config-container">
            <div class="subsection-header">
                <h4><i class="fas fa-chart-line me-2"></i>Load Profile Configuration</h4>
            </div>

            <!-- Load Method Selection -->
            <div class="mb-4">
                <h5 class="mb-3"><i class="fas fa-cog me-2"></i>Load Data Method</h5>
                <div class="row g-3">
                    <div class="col-md-4">
                        <div class="load-method-card selected" data-method="auto" onclick="selectLoadMethod('auto')">
                            <div class="text-center">
                                <div class="method-icon text-primary">
                                    <i class="fas fa-magic"></i>
                                </div>
                                <h6 class="fw-semibold">Auto-Generate</h6>
                                <p class="text-muted small mb-0">Generate typical load profile from annual energy</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="load-method-card" data-method="manual" onclick="selectLoadMethod('manual')">
                            <div class="text-center">
                                <div class="method-icon text-warning">
                                    <i class="fas fa-edit"></i>
                                </div>
                                <h6 class="fw-semibold">Manual Entry</h6>
                                <p class="text-muted small mb-0">Enter monthly energy consumption</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="load-method-card" data-method="file" onclick="selectLoadMethod('file')">
                            <div class="text-center">
                                <div class="method-icon text-success">
                                    <i class="fas fa-upload"></i>
                                </div>
                                <h6 class="fw-semibold">Upload CSV</h6>
                                <p class="text-muted small mb-0">Upload 8760-hour load data</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Load Configuration Content -->
            <div id="load-config-content">
                <!-- Will be populated based on selected method -->
            </div>

            <!-- Load Validation Results -->
            <div id="load-validation-display" style="display: none;">
                <!-- Validation results will appear here -->
            </div>

            <!-- Load Chart Display -->
            <div id="load-chart-display" style="display: none;">
                <h5 class="mb-3"><i class="fas fa-chart-area me-2"></i>Load Profile Preview</h5>
                <div class="load-chart-container" id="load-chart-container">
                    <!-- Chart will be rendered here -->
                </div>
            </div>

            <!-- Hidden fields for PySAM integration -->
            <input type="hidden" id="hidden_load_method" data-pysam-key="load_method">
            <input type="hidden" id="hidden_load_data" data-pysam-key="load_data">
            <input type="hidden" id="hidden_load_annual_energy" data-pysam-key="load_annual_energy">
            <input type="hidden" id="hidden_load_type" data-pysam-key="load_type">

            <!-- PySAM Grid Configuration Hidden Fields -->
            <input type="hidden" id="hidden_enable_interconnection_limit" data-pysam-key="grid.enable_interconnection_limit">
            <input type="hidden" id="hidden_grid_interconnection_limit_kwac" data-pysam-key="grid.grid_interconnection_limit_kwac">
            <input type="hidden" id="hidden_load_escalation" data-pysam-key="grid.load_escalation">
            <input type="hidden" id="hidden_load_step" data-pysam-key="grid.load_step">
            <input type="hidden" id="hidden_load_peak" data-pysam-key="grid.load_peak">
            <input type="hidden" id="hidden_annual_energy" data-pysam-key="grid.annual_energy">
        </div>
    `;
}

// Load method selection
function selectLoadMethod(method) {
    window.loadDataManager.currentMethod = method;

    // Update UI
    document.querySelectorAll('.load-method-card').forEach(card => {
        card.classList.remove('selected');
    });
    document.querySelector(`[data-method="${method}"]`).classList.add('selected');

    // Update content
    updateLoadConfigContent();
    updateLoadHiddenFields();
}

function updateLoadConfigContent() {
    const container = document.getElementById('load-config-content');
    const method = window.loadDataManager.currentMethod;

    switch (method) {
        case 'auto':
            container.innerHTML = createAutoGenerateContent();
            break;
        case 'manual':
            container.innerHTML = createManualEntryContent();
            break;
        case 'file':
            container.innerHTML = createFileUploadContent();
            break;
    }

    setupLoadEventListeners();
}

function createAutoGenerateContent() {
    return `
        <div class="rate-input-group">
            <h5 class="mb-3"><i class="fas fa-magic me-2"></i>Auto-Generation Settings</h5>
            <div class="row g-3">
                <div class="col-md-6">
                    <label for="load-annual-energy" class="form-label fw-semibold">
                        Annual Energy Consumption (kWh)
                    </label>
                    <input type="number" id="load-annual-energy" class="form-control"
                           value="10000" min="0" step="100"
                           onchange="updateAutoGeneration()">
                    <div class="form-text">Typical residential: 8,000-15,000 kWh/year</div>
                </div>
                <div class="col-md-6">
                    <label for="load-type-select" class="form-label fw-semibold">Load Type</label>
                    <select id="load-type-select" class="form-control" onchange="updateAutoGeneration()">
                        <option value="residential">Residential</option>
                        <option value="commercial">Commercial</option>
                        <option value="industrial">Industrial</option>
                    </select>
                    <div class="form-text">Determines hourly usage patterns</div>
                </div>
                <div class="col-12">
                    <button type="button" class="btn btn-primary" onclick="generateAutoLoad()">
                        <i class="fas fa-chart-line me-2"></i>Generate Load Profile
                    </button>
                </div>
            </div>
        </div>
    `;
}

function createManualEntryContent() {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

    return `
        <div class="rate-input-group">
            <h5 class="mb-3"><i class="fas fa-edit me-2"></i>Monthly Energy Consumption</h5>
            <div class="row g-3 mb-3">
                <div class="col-md-6">
                    <label for="load-type-select" class="form-label fw-semibold">Load Type</label>
                    <select id="load-type-select" class="form-control">
                        <option value="residential">Residential</option>
                        <option value="commercial">Commercial</option>
                        <option value="industrial">Industrial</option>
                    </select>
                </div>
                <div class="col-md-6">
                    <label class="form-label fw-semibold">Quick Fill</label>
                    <div class="d-flex gap-2">
                        <button type="button" class="btn btn-outline-secondary btn-sm" onclick="fillMonthlyAverage()">
                            Average
                        </button>
                        <button type="button" class="btn btn-outline-secondary btn-sm" onclick="clearMonthlyData()">
                            Clear All
                        </button>
                    </div>
                </div>
            </div>
            <div class="monthly-input-grid">
                ${months.map((month, index) => `
                    <div class="month-input-card">
                        <label for="month-${index + 1}-energy" class="form-label fw-semibold">${month}</label>
                        <div class="input-group">
                            <input type="number" id="month-${index + 1}-energy" class="form-control"
                                   min="0" step="10" placeholder="0" onchange="updateMonthlyTotal()">
                            <span class="input-group-text">kWh</span>
                        </div>
                    </div>
                `).join('')}
            </div>
            <div class="mt-3 d-flex justify-content-between align-items-center">
                <div>
                    <strong>Total Annual: <span id="monthly-total">0</span> kWh</strong>
                </div>
                <button type="button" class="btn btn-primary" onclick="generateManualLoad()">
                    <i class="fas fa-chart-line me-2"></i>Generate Load Profile
                </button>
            </div>
        </div>
    `;
}

function createFileUploadContent() {
    return `
        <div class="rate-input-group">
            <h5 class="mb-3"><i class="fas fa-upload me-2"></i>Upload Load Data File</h5>
            <div class="upload-area" id="upload-area"
                 ondrop="handleFileDrop(event)"
                 ondragover="handleDragOver(event)"
                 ondragleave="handleDragLeave(event)">
                <div class="text-center">
                    <i class="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
                    <h6>Drop CSV file here or click to browse</h6>
                    <p class="text-muted mb-3">File must contain 8760 hourly load values (kWh)</p>
                    <input type="file" id="load-file-input" accept=".csv" style="display: none;" onchange="handleFileSelect(event)">
                    <button type="button" class="btn btn-outline-primary" onclick="document.getElementById('load-file-input').click()">
                        <i class="fas fa-folder-open me-2"></i>Browse Files
                    </button>
                </div>
            </div>

            <div id="upload-progress-container" style="display: none;">
                <div class="upload-progress">
                    <div class="upload-progress-bar" id="upload-progress-bar" style="width: 0%;"></div>
                </div>
                <div class="text-center">
                    <small class="text-muted" id="upload-status">Uploading...</small>
                </div>
            </div>

            <div class="mt-3">
                <h6>File Format Requirements:</h6>
                <ul class="small text-muted">
                    <li>CSV format with single column of hourly load values</li>
                    <li>Exactly 8760 rows (365 days × 24 hours)</li>
                    <li>Values in kWh (positive numbers)</li>
                    <li>Optional header row will be detected automatically</li>
                </ul>
            </div>
        </div>
    `;
}

function setupLoadEventListeners() {
    // Auto-generation listeners are set up via onchange attributes
    // Manual entry listeners are set up via onchange attributes
    // File upload listeners are set up via on* attributes
}

// Auto-generation functions
function updateAutoGeneration() {
    updateLoadHiddenFields();
}

function generateAutoLoad() {
    const annualEnergy = parseFloat(document.getElementById('load-annual-energy').value) || 10000;
    const loadType = document.getElementById('load-type-select').value || 'residential';

    showNotification('Generating load profile...', 'info');

    try {
        const profile = window.loadDataManager.generateHourlyProfile(annualEnergy, loadType);
        window.loadDataManager.loadData = profile;
        window.loadDataManager.validationResult = { valid: true, message: 'Auto-generated profile' };

        displayLoadValidation({ valid: true, message: 'Load profile generated successfully' });
        displayLoadChart(profile);
        updateLoadHiddenFields();

        showNotification('Load profile generated successfully!', 'success');
    } catch (error) {
        showNotification('Failed to generate load profile: ' + error.message, 'error');
    }
}

// Manual entry functions
function updateMonthlyTotal() {
    let total = 0;
    for (let i = 1; i <= 12; i++) {
        const input = document.getElementById(`month-${i}-energy`);
        total += parseFloat(input?.value) || 0;
    }

    const totalSpan = document.getElementById('monthly-total');
    if (totalSpan) {
        totalSpan.textContent = total.toLocaleString();
    }
}

function fillMonthlyAverage() {
    const annualEnergy = prompt('Enter total annual energy consumption (kWh):', '10000');
    if (annualEnergy && !isNaN(annualEnergy)) {
        const monthlyAverage = Math.round(parseFloat(annualEnergy) / 12);
        for (let i = 1; i <= 12; i++) {
            const input = document.getElementById(`month-${i}-energy`);
            if (input) input.value = monthlyAverage;
        }
        updateMonthlyTotal();
    }
}

function clearMonthlyData() {
    for (let i = 1; i <= 12; i++) {
        const input = document.getElementById(`month-${i}-energy`);
        if (input) input.value = '';
    }
    updateMonthlyTotal();
}

function generateManualLoad() {
    const monthlyData = [];
    for (let i = 1; i <= 12; i++) {
        const input = document.getElementById(`month-${i}-energy`);
        monthlyData.push(parseFloat(input?.value) || 0);
    }

    const total = monthlyData.reduce((sum, val) => sum + val, 0);
    if (total === 0) {
        showNotification('Please enter monthly energy values first', 'warning');
        return;
    }

    const loadType = document.getElementById('load-type-select').value || 'residential';

    showNotification('Generating load profile from monthly data...', 'info');

    try {
        const profile = window.loadDataManager.generateFromMonthlyData(monthlyData, loadType);
        window.loadDataManager.loadData = profile;
        window.loadDataManager.validationResult = { valid: true, message: 'Generated from monthly data' };

        displayLoadValidation({ valid: true, message: 'Load profile generated from monthly data' });
        displayLoadChart(profile);
        updateLoadHiddenFields();

        showNotification('Load profile generated successfully!', 'success');
    } catch (error) {
        showNotification('Failed to generate load profile: ' + error.message, 'error');
    }
}

// File upload functions
function handleDragOver(event) {
    event.preventDefault();
    event.currentTarget.classList.add('drag-over');
}

function handleDragLeave(event) {
    event.currentTarget.classList.remove('drag-over');
}

function handleFileDrop(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('drag-over');

    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processLoadFile(files[0]);
    }
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processLoadFile(file);
    }
}

async function processLoadFile(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showNotification('Please select a CSV file', 'error');
        return;
    }

    showUploadProgress(0);

    try {
        const result = await window.loadDataManager.uploadFile(file);

        if (result.success) {
            hideUploadProgress();
            displayLoadValidation(result.data.validation);

            if (result.data.validation.valid) {
                displayLoadChart(result.data.data);
                showNotification('Load data uploaded and validated successfully!', 'success');
            } else {
                showNotification('Load data validation failed: ' + result.data.validation.error, 'error');
            }

            updateLoadHiddenFields();
        } else {
            hideUploadProgress();
            showNotification('Upload failed: ' + result.error, 'error');
        }
    } catch (error) {
        hideUploadProgress();
        showNotification('Upload error: ' + error.message, 'error');
    }
}

function showUploadProgress(percent) {
    const container = document.getElementById('upload-progress-container');
    const bar = document.getElementById('upload-progress-bar');
    const status = document.getElementById('upload-status');

    if (container) container.style.display = 'block';
    if (bar) bar.style.width = percent + '%';
    if (status) status.textContent = percent < 100 ? 'Uploading...' : 'Processing...';
}

function hideUploadProgress() {
    const container = document.getElementById('upload-progress-container');
    if (container) container.style.display = 'none';
}

// Display functions
function displayLoadValidation(result) {
    const container = document.getElementById('load-validation-display');
    if (!container) return;

    const className = result.valid ? 'success' : 'error';
    const icon = result.valid ? 'fas fa-check-circle' : 'fas fa-exclamation-circle';

    container.innerHTML = `
        <div class="load-validation-result ${className}">
            <div class="d-flex align-items-center">
                <i class="${icon} me-2"></i>
                <div class="flex-grow-1">
                    <strong>${result.valid ? 'Validation Passed' : 'Validation Failed'}</strong>
                    <div class="mt-1">${result.message || result.error || 'Unknown status'}</div>
                </div>
            </div>
        </div>
    `;

    container.style.display = 'block';
}

function displayLoadChart(data) {
    const container = document.getElementById('load-chart-display');
    if (!container) return;

    window.loadDataManager.destroyChart();
    window.loadDataManager.createChart('load-chart-container', data);

    container.style.display = 'block';
}

function updateLoadHiddenFields() {
    const method = window.loadDataManager.currentMethod;
    const formData = window.loadDataManager.getFormData();

    // Update hidden fields
    const methodField = document.getElementById('hidden_load_method');
    if (methodField) methodField.value = method;

    const dataField = document.getElementById('hidden_load_data');
    if (dataField && formData.load_data) {
        dataField.value = JSON.stringify(formData.load_data);
    }

    const annualField = document.getElementById('hidden_load_annual_energy');
    if (annualField && formData.load_annual_energy) {
        annualField.value = formData.load_annual_energy;
    }

    const typeField = document.getElementById('hidden_load_type');
    if (typeField && formData.load_type) {
        typeField.value = formData.load_type;
    }

    // Update the new PySAM grid fields - dynamically get values
    const gridFields = [
        'enable_interconnection_limit',
        'grid_interconnection_limit_kwac',
        'load_escalation',
        'load_step',
        'load_peak',
        'annual_energy'
    ];

    gridFields.forEach(fieldId => {
        const field = document.getElementById(fieldId);
        const hiddenField = document.getElementById(`hidden_${fieldId}`);

        if (field && hiddenField) {
            hiddenField.value = field.value;
        } else if (field && !hiddenField) {
            // Create hidden field if it doesn't exist
            const hidden = document.createElement('input');
            hidden.type = 'hidden';
            hidden.id = `hidden_${fieldId}`;
            hidden.name = `hidden_${fieldId}`;
            hidden.value = field.value;
            const container = document.getElementById('load-config-container');
            if (container) container.appendChild(hidden);
        }
    });
}

function initLoadSection() {
    console.log("Initializing load section...");

    window.loadDataManager.initialize();
    updateLoadConfigContent();

    // Apply defaults from form config to grid fields
    const gridFields = formConfig.filter(f => f.section === "Load");
    gridFields.forEach(configField => {
        if (configField.default_value !== undefined) {
            const element = document.getElementById(configField.key);
            if (element) {
                if (element.type === 'checkbox') {
                    element.checked = configField.default_value === 'true' || configField.default_value === true;
                } else if (element.tagName === 'SELECT') {
                    element.value = configField.default_value;
                } else {
                    element.value = configField.default_value;
                }
            }
        }
    });

    updateLoadHiddenFields();

    // Set up initial state
    generateAutoLoad(); // Generate default profile

    // Set up event listeners for PySAM grid fields
    const gridFieldIds = [
        'enable_interconnection_limit',
        'grid_interconnection_limit_kwac',
        'load_escalation',
        'load_step',
        'load_peak',
        'annual_energy'
    ];

    gridFieldIds.forEach(fieldId => {
        const field = document.getElementById(fieldId);
        if (field) {
            field.addEventListener('change', updateLoadHiddenFields);
            field.addEventListener('input', updateLoadHiddenFields);
        }
    });
}

// Global FCAS state
let fcasState = {
    enabled: false,
    region: 'NSW1',
    batterySize: 100,
    participationRate: 85,
    services: {
        'fcas_enable_fast_raise': true,
        'fcas_enable_fast_lower': true,
        'fcas_enable_slow_raise': true,
        'fcas_enable_slow_lower': true,
        'fcas_enable_delayed_raise': false,
        'fcas_enable_delayed_lower': false,
        'fcas_enable_raise_regulation': false,
        'fcas_enable_lower_regulation': false
    },
    forecastMethod: 'ML',
    databasePath: 'FCAS/nem_data_final.db'
};

// Load form configuration
async function loadFormConfig() {
    try {
        const response = await fetch('/static/js/form_config.json?' + new Date().getTime());
        if (!response.ok) {
            throw new Error('HTTP error! status: ' + response.status);
        }
        const json = await response.json();
        console.log("Config loaded successfully with", json.length, "items");
        formConfig = json;
        return json;
    } catch (err) {
        console.error("Failed to load config:", err);
        return [];
    }
}

// Auto-detect FCAS region from latitude/longitude
function detectFCASRegion(lat, lon) {
    if (!lat || !lon) return 'NSW1';

    if (lat < -37.5 && lon > 140) return 'VIC1';
    if (lat < -34.5 && lon < 138.5) return 'SA1';
    if (lat > -28 && lon > 150) return 'QLD1';
    if (lat < -31.5 && lon < 116) return 'WA1';
    if (lat < -35 && lon > 147) return 'NSW1';
    if (lat > -23 && lon > 130 && lon < 135) return 'NT1';
    if (lat < -42) return 'TAS1';

    return 'NSW1';
}

// Calculate participation rate based on battery size
function calculateParticipationRate(batteryKWh) {
    if (!batteryKWh || batteryKWh < 1) return 0;

    if (batteryKWh < 50) return Math.min(50, batteryKWh * 1.2);
    if (batteryKWh < 100) return Math.min(70, 50 + (batteryKWh - 50) * 0.6);
    if (batteryKWh < 500) return Math.min(85, 70 + (batteryKWh - 100) * 0.04);

    return Math.min(95, 85 + Math.log10(batteryKWh / 500) * 5);
}

// Estimate FCAS revenue potential
function estimateFCASRevenue() {
    if (!fcasState.enabled || fcasState.batterySize < 1) return 0;

    const serviceRates = {
        'fcas_enable_fast_raise': 45000,
        'fcas_enable_fast_lower': 42000,
        'fcas_enable_slow_raise': 35000,
        'fcas_enable_slow_lower': 32000,
        'fcas_enable_delayed_raise': 25000,
        'fcas_enable_delayed_lower': 23000,
        'fcas_enable_raise_regulation': 18000,
        'fcas_enable_lower_regulation': 16000
    };

    const regionMultipliers = {
        'NSW1': 1.0,
        'VIC1': 0.95,
        'QLD1': 1.05,
        'SA1': 1.15,
        'TAS1': 0.85,
        'WA1': 0.7
    };

    let totalRevenue = 0;
    const batteryMW = fcasState.batterySize / 1000;
    const participationFactor = fcasState.participationRate / 100;
    const regionFactor = regionMultipliers[fcasState.region] || 1.0;

    Object.entries(fcasState.services).forEach(([service, enabled]) => {
        if (enabled && serviceRates[service]) {
            totalRevenue += serviceRates[service] * batteryMW * participationFactor * regionFactor;
        }
    });

    return Math.round(totalRevenue);
}

// Create FCAS configuration section
function createFCASSection() {
    return `
        <div class="fcas-section" id="fcas-config-container">
            <div class="subsection-header">
                <h4><i class="fas fa-bolt me-2"></i>FCAS (Frequency Control Ancillary Services)</h4>
            </div>

            <div class="fcas-auto-config mb-4">
                <div class="d-flex align-items-center justify-content-between">
                    <div>
                        <h5 class="mb-1"><i class="fas fa-magic me-2"></i>Enhanced FCAS Revenue Forecasting</h5>
                        <p class="text-muted mb-0 fs-6">ML-based forecasting for Australian NEM FCAS markets</p>
                    </div>
                    <div class="form-check form-switch">
                        <input class="form-check-input"
                               type="checkbox"
                               id="fcas-enable"
                               data-pysam-key="fcas_enabled"
                               checked>
                        <label class="form-check-label fw-semibold" for="fcas-enable">
                            Enable FCAS
                        </label>
                    </div>
                </div>
            </div>

            <div id="fcas-configuration" class="space-y-6">
                <div class="fcas-auto-config">
                    <h5 class="mb-3"><i class="fas fa-cog me-2"></i>Auto-Configuration</h5>
                    <div class="row g-3">
                        <div class="col-md-4">
                            <label for="fcas-region" class="form-label fw-semibold">
                                NEM Region <i class="fas fa-info-circle text-muted" title="Auto-detected from project location"></i>
                            </label>
                            <select id="fcas-region" class="form-control" data-pysam-key="fcas_region">
                                <option value="NSW1">NSW1 - New South Wales</option>
                                <option value="VIC1">VIC1 - Victoria</option>
                                <option value="QLD1">QLD1 - Queensland</option>
                                <option value="SA1">SA1 - South Australia</option>
                                <option value="TAS1">TAS1 - Tasmania</option>
                            </select>
                        </div>
                        <div class="col-md-4">
                            <label for="fcas-participation-rate" class="form-label fw-semibold">
                                Participation Rate (%) <i class="fas fa-info-circle text-muted" title="Auto-calculated from battery size"></i>
                            </label>
                            <input type="number"
                                   id="fcas-participation-rate"
                                   class="form-control"
                                   data-pysam-key="fcas_participation_rate"
                                   min="0" max="100" step="1" value="85" readonly>
                        </div>
                        <div class="col-md-4">
                            <label for="fcas-battery-reserve" class="form-label fw-semibold">
                                Battery Reserve (%)
                            </label>
                            <input type="number"
                                   id="fcas-battery-reserve"
                                   class="form-control"
                                   data-pysam-key="fcas_battery_reserve"
                                   min="0" max="50" step="5" value="20">
                        </div>
                    </div>
                </div>

                <div class="rate-input-group">
                    <h5 class="mb-3"><i class="fas fa-list-check me-2"></i>FCAS Services Selection</h5>
                    <div class="fcas-service-grid" id="fcas-services-grid">
                        <!-- Services will be populated by JavaScript -->
                    </div>
                </div>

                <div class="fcas-revenue-estimate" id="fcas-revenue-display">
                    <h5 class="mb-2"><i class="fas fa-chart-line me-2"></i>Estimated Annual FCAS Revenue</h5>
                    <div class="fs-2 fw-bold text-success" id="fcas-revenue-amount">$0</div>
                    <small class="text-muted">Based on current configuration and historical market data</small>
                </div>

                <div class="collapse" id="fcas-advanced-config">
                    <div class="rate-input-group">
                        <h5 class="mb-3"><i class="fas fa-cogs me-2"></i>Advanced Configuration</h5>
                        <div class="row g-3">
                            <div class="col-md-6">
                                <label for="fcas-forecast-method" class="form-label fw-semibold">Forecast Method</label>
                                <select id="fcas-forecast-method"
                                        class="form-control"
                                        data-pysam-key="fcas_forecast_method">
                                    <option value="ML">Machine Learning (Recommended)</option>
                                    <option value="historical">Historical Average</option>
                                    <option value="conservative">Conservative Estimate</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label for="fcas-database-path" class="form-label fw-semibold">Database Path</label>
                                <input type="text"
                                       id="fcas-database-path"
                                       class="form-control"
                                       data-pysam-key="fcas_db_path"
                                       value="FCAS/nem_data_final.db" readonly>
                            </div>
                        </div>
                        <div class="row g-3 mt-2">
                            <div class="col-md-6">
                                <label for="fcas-fast-premium" class="form-label fw-semibold">Fast Services Premium ($/MW/5min)</label>
                                <input type="number"
                                       id="fcas-fast-premium"
                                       class="form-control"
                                       data-pysam-key="fcas_fast_premium"
                                       min="0" step="0.01" value="0">
                            </div>
                            <div class="col-md-6">
                                <label for="fcas-regulation-premium" class="form-label fw-semibold">Regulation Premium ($/MW/5min)</label>
                                <input type="number"
                                       id="fcas-regulation-premium"
                                       class="form-control"
                                       data-pysam-key="fcas_regulation_premium"
                                       min="0" step="0.01" value="0">
                            </div>
                        </div>
                    </div>
                </div>

                <div class="text-center">
                    <button type="button" class="btn btn-outline-secondary btn-sm"
                            data-bs-toggle="collapse" data-bs-target="#fcas-advanced-config">
                        <i class="fas fa-chevron-down me-2"></i>Advanced Configuration
                    </button>
                </div>
            </div>
        </div>
    `;
}

// Create FCAS services grid
function renderFCASServices() {
    const servicesGrid = document.getElementById('fcas-services-grid');
    if (!servicesGrid) return;

    const services = [
        {
            key: 'fcas_enable_fast_raise',
            name: 'Fast Raise (6s)',
            description: 'Rapid frequency response for under-frequency events',
            recommended: true,
            icon: 'fas fa-arrow-up'
        },
        {
            key: 'fcas_enable_fast_lower',
            name: 'Fast Lower (6s)',
            description: 'Rapid frequency response for over-frequency events',
            recommended: true,
            icon: 'fas fa-arrow-down'
        },
        {
            key: 'fcas_enable_slow_raise',
            name: 'Slow Raise (60s)',
            description: 'Sustained frequency response for under-frequency events',
            recommended: true,
            icon: 'fas fa-chevron-up'
        },
        {
            key: 'fcas_enable_slow_lower',
            name: 'Slow Lower (60s)',
            description: 'Sustained frequency response for over-frequency events',
            recommended: true,
            icon: 'fas fa-chevron-down'
        },
        {
            key: 'fcas_enable_delayed_raise',
            name: 'Delayed Raise (5min)',
            description: 'Extended frequency response for under-frequency events',
            recommended: false,
            icon: 'fas fa-clock'
        },
        {
            key: 'fcas_enable_delayed_lower',
            name: 'Delayed Lower (5min)',
            description: 'Extended frequency response for over-frequency events',
            recommended: false,
            icon: 'fas fa-clock'
        },
        {
            key: 'fcas_enable_raise_regulation',
            name: 'Raise Regulation',
            description: 'Continuous regulation for small frequency deviations',
            recommended: false,
            icon: 'fas fa-wave-square'
        },
        {
            key: 'fcas_enable_lower_regulation',
            name: 'Lower Regulation',
            description: 'Continuous regulation for small frequency deviations',
            recommended: false,
            icon: 'fas fa-wave-square'
        }
    ];

    servicesGrid.innerHTML = services.map(service => `
        <div class="fcas-service-card ${fcasState.services[service.key] ? 'enabled' : ''}"
             data-service="${service.key}">
            <div class="d-flex align-items-start justify-content-between mb-2">
                <div class="d-flex align-items-center">
                    <i class="${service.icon} me-2 text-primary"></i>
                    <strong>${service.name}</strong>
                </div>
                <div class="form-check form-switch">
                    <input class="form-check-input fcas-service-toggle"
                           type="checkbox"
                           id="${service.key}"
                           data-pysam-key="${service.key}"
                           ${fcasState.services[service.key] ? 'checked' : ''}>
                </div>
            </div>
            <p class="text-muted small mb-2">${service.description}</p>
            ${service.recommended ? '<span class="badge bg-success">Recommended</span>' : '<span class="badge bg-secondary">Optional</span>'}
        </div>
    `).join('');

    document.querySelectorAll('.fcas-service-toggle').forEach(toggle => {
        toggle.addEventListener('change', (e) => {
            const serviceKey = e.target.id;
            fcasState.services[serviceKey] = e.target.checked;

            const card = e.target.closest('.fcas-service-card');
            card.classList.toggle('enabled', e.target.checked);

            updateFCASRevenue();
            updateFCASHiddenFields();
        });
    });
}

// Update FCAS revenue estimate
function updateFCASRevenue() {
    const revenueAmount = document.getElementById('fcas-revenue-amount');
    if (revenueAmount) {
        const revenue = estimateFCASRevenue();
        revenueAmount.textContent = `${revenue.toLocaleString()}`;
    }
}

// Initialize FCAS section
function initFCASSection() {
    console.log("Initializing FCAS section...");

    createHiddenFCASFields();
    renderFCASServices();
    setupFCASEventListeners();

    const fcasToggle = document.getElementById('fcas-enable');
    const fcasConfiguration = document.getElementById('fcas-configuration');

    if (fcasToggle && fcasConfiguration) {
        const currentToggleState = fcasToggle.checked;

        if (!fcasToggle.hasAttribute('data-user-configured')) {
            const batteryCapacity = document.getElementById('input__batt_computed_bank_capacity')?.value || 0;
            const batteryEnabled = document.getElementById('input__en_batt')?.value === 'yes';

            if (batteryEnabled && parseFloat(batteryCapacity) > 0) {
                fcasToggle.checked = true;
                fcasConfiguration.style.display = 'block';
                fcasState.enabled = true;
                console.log("[FCAS] Auto-enabled due to battery presence");
            } else {
                fcasToggle.checked = false;
                fcasConfiguration.style.display = 'none';
                fcasState.enabled = false;
                console.log("[FCAS] Auto-disabled due to no battery");
            }
        } else {
            fcasState.enabled = currentToggleState;
            fcasConfiguration.style.display = currentToggleState ? 'block' : 'none';
            console.log("[FCAS] Preserved user setting:", currentToggleState);
        }
    }

    setTimeout(() => {
        const latInput = document.getElementById('latitude');
        const lonInput = document.getElementById('longitude');
        if (latInput && lonInput && latInput.value && lonInput.value) {
            const detectedRegion = detectFCASRegion(parseFloat(latInput.value), parseFloat(lonInput.value));
            fcasState.region = detectedRegion;
            const regionSelect = document.getElementById('fcas-region');
            if (regionSelect && !regionSelect.hasAttribute('data-user-configured')) {
                regionSelect.value = detectedRegion;
            }
            updateFCASRevenue();
        }
    }, 1000);

    setTimeout(() => {
        const batteryInput = document.getElementById('input__batt_computed_bank_capacity');
        if (batteryInput && batteryInput.value) {
            fcasState.batterySize = parseFloat(batteryInput.value) || 100;
            const participationRate = calculateParticipationRate(fcasState.batterySize);
            fcasState.participationRate = participationRate;
            const participationInput = document.getElementById('fcas-participation-rate');
            if (participationInput) {
                participationInput.value = participationRate;
            }
            updateFCASRevenue();
        }
    }, 1000);

    updateFCASRevenue();
    updateFCASHiddenFields();
}

// Enhanced FCAS event listener setup
function setupFCASEventListeners() {
    const fcasEnable = document.getElementById('fcas-enable');
    if (fcasEnable) {
        fcasEnable.addEventListener('change', (e) => {
            const isEnabled = e.target.checked;
            fcasState.enabled = isEnabled;

            fcasEnable.setAttribute('data-user-configured', 'true');

            const configSection = document.getElementById('fcas-configuration');
            if (configSection) {
                configSection.style.display = isEnabled ? 'block' : 'none';
            }

            updateFCASRevenue();
            updateFCASHiddenFields();

            console.log(`[FCAS] User toggled: ${isEnabled ? 'ENABLED' : 'DISABLED'}`);
        });
    }

    const fcasRegion = document.getElementById('fcas-region');
    if (fcasRegion) {
        fcasRegion.addEventListener('change', (e) => {
            fcasState.region = e.target.value;
            fcasRegion.setAttribute('data-user-configured', 'true');
            updateFCASRevenue();
            updateFCASHiddenFields();
            console.log(`[FCAS] User set region: ${e.target.value}`);
        });
    }

    document.addEventListener('change', (e) => {
        if (e.target.classList.contains('fcas-service-toggle')) {
            const serviceKey = e.target.id;
            fcasState.services[serviceKey] = e.target.checked;

            e.target.setAttribute('data-user-configured', 'true');

            const card = e.target.closest('.fcas-service-card');
            if (card) {
                card.classList.toggle('enabled', e.target.checked);
            }

            updateFCASRevenue();
            updateFCASHiddenFields();
            console.log(`[FCAS] User toggled service ${serviceKey}: ${e.target.checked}`);
        }
    });

    const batteryReserve = document.getElementById('fcas-battery-reserve');
    if (batteryReserve) {
        batteryReserve.addEventListener('input', (e) => {
            e.target.setAttribute('data-user-configured', 'true');
            updateFCASHiddenFields();
        });
    }

    const trackedInputs = [
        'fcas-forecast-method',
        'fcas-fast-premium',
        'fcas-regulation-premium'
    ];

    trackedInputs.forEach(inputId => {
        const input = document.getElementById(inputId);
        if (input) {
            input.addEventListener('change', (e) => {
                e.target.setAttribute('data-user-configured', 'true');
                updateFCASHiddenFields();
            });
        }
    });

    document.addEventListener('input', (e) => {
        if (e.target.id === 'input__batt_computed_bank_capacity') {
            fcasState.batterySize = parseFloat(e.target.value) || 100;
            const participationRate = calculateParticipationRate(fcasState.batterySize);
            fcasState.participationRate = participationRate;

            const participationInput = document.getElementById('fcas-participation-rate');
            if (participationInput && !participationInput.hasAttribute('data-user-configured')) {
                participationInput.value = participationRate;
            }

            updateFCASRevenue();
            updateFCASHiddenFields();
        }
    });

    document.addEventListener('change', (e) => {
        if (e.target.id === 'input__en_batt') {
            const batteryEnabled = e.target.value === 'yes';
            const fcasToggle = document.getElementById('fcas-enable');

            if (!batteryEnabled && fcasToggle && fcasToggle.checked) {
                if (!fcasToggle.hasAttribute('data-user-configured')) {
                    fcasToggle.checked = false;
                    fcasToggle.dispatchEvent(new Event('change'));
                    console.log("[FCAS] Auto-disabled due to battery disable");
                } else {
                    console.log("[FCAS] Battery disabled but preserving user FCAS setting");
                }
            }
        }
    });

    document.addEventListener('input', (e) => {
        if (e.target.id === 'latitude' || e.target.id === 'longitude') {
            setTimeout(() => {
                const latInput = document.getElementById('latitude');
                const lonInput = document.getElementById('longitude');
                const fcasRegionSelect = document.getElementById('fcas-region');

                if (latInput && lonInput && latInput.value && lonInput.value && fcasRegionSelect) {
                    if (!fcasRegionSelect.hasAttribute('data-user-configured')) {
                        const detectedRegion = detectFCASRegion(parseFloat(latInput.value), parseFloat(lonInput.value));
                        fcasState.region = detectedRegion;
                        fcasRegionSelect.value = detectedRegion;
                        updateFCASRevenue();
                        updateFCASHiddenFields();
                        console.log("[FCAS] Auto-updated region to:", detectedRegion);
                    }
                }
            }, 500);
        }
    });
}

// Enhanced hidden field updates with explicit enable state
function updateFCASHiddenFields() {
    const updates = {
        'hidden_fcas_enabled': fcasState.enabled ? 'Yes' : 'No',
        'hidden_fcas_region': fcasState.enabled ? fcasState.region : 'DISABLED',
        'hidden_fcas_participation_rate': fcasState.enabled ? fcasState.participationRate : 0,
        'hidden_fcas_db_path': fcasState.enabled ? fcasState.databasePath : '',
        'hidden_fcas_battery_reserve': fcasState.enabled ? (document.getElementById('fcas-battery-reserve')?.value || 20) : 0,
        'hidden_fcas_forecast_method': fcasState.enabled ? fcasState.forecastMethod : 'disabled',
        'hidden_fcas_fast_premium': fcasState.enabled ? (document.getElementById('fcas-fast-premium')?.value || 0) : 0,
        'hidden_fcas_regulation_premium': fcasState.enabled ? (document.getElementById('fcas-regulation-premium')?.value || 0) : 0
    };

    Object.entries(fcasState.services).forEach(([service, enabled]) => {
        updates[`hidden_${service}`] = (fcasState.enabled && enabled) ? 'Yes' : 'No';
    });

    Object.entries(updates).forEach(([fieldId, value]) => {
        const field = document.getElementById(fieldId);
        if (field) {
            field.value = value;
        }
    });

    console.log("[FCAS] Hidden fields updated, enabled:", fcasState.enabled);
}

// Add explicit fcas_enabled hidden field creation
function createHiddenFCASFields() {
    const container = document.getElementById('fcas-config-container');
    if (!container) return;

    const fcasFields = [
        'fcas_enabled',
        'fcas_region', 'fcas_participation_rate', 'fcas_db_path', 'fcas_battery_reserve',
        'fcas_enable_fast_raise', 'fcas_enable_fast_lower', 'fcas_enable_slow_raise',
        'fcas_enable_slow_lower', 'fcas_enable_delayed_raise', 'fcas_enable_delayed_lower',
        'fcas_enable_raise_regulation', 'fcas_enable_lower_regulation', 'fcas_forecast_method',
        'fcas_fast_premium', 'fcas_regulation_premium'
    ];

    fcasFields.forEach(fieldName => {
        const hiddenFieldId = `hidden_${fieldName}`;
        if (!document.getElementById(hiddenFieldId)) {
            const hiddenField = document.createElement('input');
            hiddenField.type = 'hidden';
            hiddenField.id = hiddenFieldId;
            hiddenField.dataset.pysamModule = 'Enhanced_FCAS';
            hiddenField.dataset.pysamKey = fieldName;
            container.appendChild(hiddenField);
        }
    });
}

// ========== EXISTING TARIFF FUNCTIONALITY ==========

let currentTariffType = 'Flat';
let currentTOUPeriodCount = 2;
let touWeekdaySchedule = Array(24).fill(1);
let touWeekendSchedule = Array(24).fill(1);
let touRates = [0.12, 0.08];

function createTariffSection() {
    return `
        <div class="tariff-section" id="tariff-config-container">
            <div class="subsection-header">
                <h4><i class="fas fa-plug me-2"></i>Network Tariff Configuration</h4>
            </div>
            <div id="tariff-controls" class="space-y-6"></div>
            <div id="tariff-config-details" class="space-y-6"></div>
        </div>
    `;
}

function initTariffSection() {
    console.log("Initializing tariff section...");

    createHiddenPYSAMFields();
    renderTariffTypeSelector();
    renderFixedChargeField();
    renderFlatRateUI();
    setupTariffEventListeners();

    const tariffFields = formConfig.filter(f => f.section === "Network Configuration");
    tariffFields.forEach(field => {
        if (field.default_value !== undefined) {
            setTimeout(() => {
                const input = document.getElementById(`input__${field.key}`);
                if (input) {
                    if (input.type === 'checkbox') {
                        input.checked = field.default_value === 'true' || field.default_value === true;
                    } else if (input.tagName === 'SELECT') {
                        for (let i = 0; i < input.options.length; i++) {
                            if (input.options[i].value === field.default_value.toString()) {
                                input.selectedIndex = i;
                                break;
                            }
                        }
                    } else if (input.type !== 'hidden') {
                        input.value = field.default_value;
                    }
                }
            }, 0);
        }
    });
}

function createHiddenPYSAMFields() {
    const container = document.getElementById('tariff-config-container');
    if (!container) return;

    if (!document.getElementById('monthly_fixed_charge')) {
        const fixedChargeField = document.createElement('input');
        fixedChargeField.type = 'hidden';
        fixedChargeField.id = 'monthly_fixed_charge';
        fixedChargeField.dataset.pysamModule = 'UtilityRate5';
        fixedChargeField.dataset.pysamKey = 'monthly_fixed_charge';
        container.appendChild(fixedChargeField);
    }

    if (!document.getElementById('ur_ec_sched_weekday')) {
        const weekdayField = document.createElement('input');
        weekdayField.type = 'hidden';
        weekdayField.id = 'ur_ec_sched_weekday';
        weekdayField.dataset.pysamModule = 'UtilityRate5';
        weekdayField.dataset.pysamKey = 'ur_ec_sched_weekday';
        container.appendChild(weekdayField);
    }

    if (!document.getElementById('ur_ec_sched_weekend')) {
        const weekendField = document.createElement('input');
        weekendField.type = 'hidden';
        weekendField.id = 'ur_ec_sched_weekend';
        weekendField.dataset.pysamModule = 'UtilityRate5';
        weekendField.dataset.pysamKey = 'ur_ec_sched_weekend';
        container.appendChild(weekendField);
    }

    if (!document.getElementById('ur_ec_tou_mat')) {
        const touMatField = document.createElement('input');
        touMatField.type = 'hidden';
        touMatField.id = 'ur_ec_tou_mat';
        touMatField.dataset.pysamModule = 'UtilityRate5';
        touMatField.dataset.pysamKey = 'ur_ec_tou_mat';
        container.appendChild(touMatField);
    }

    if (!document.getElementById('ur_dc_flat_mat')) {
        const demandFlatField = document.createElement('input');
        demandFlatField.type = 'hidden';
        demandFlatField.id = 'ur_dc_flat_mat';
        demandFlatField.dataset.pysamModule = 'UtilityRate5';
        demandFlatField.dataset.pysamKey = 'ur_dc_flat_mat';
        container.appendChild(demandFlatField);
    }

    if (!document.getElementById('ur_ec_export_mat')) {
        const exportMatField = document.createElement('input');
        exportMatField.type = 'hidden';
        exportMatField.id = 'ur_ec_export_mat';
        exportMatField.dataset.pysamModule = 'UtilityRate5';
        exportMatField.dataset.pysamKey = 'ur_ec_export_mat';
        container.appendChild(exportMatField);
    }
}

function renderTariffTypeSelector() {
    const container = document.getElementById('tariff-controls');
    if (!container) return;

    container.innerHTML = `
        <div class="mb-4">
            <label for="tariff-type-select" class="form-label fw-semibold">Tariff Type</label>
            <select id="tariff-type-select" class="form-control">
                <option value="Flat">Flat Rate</option>
                <option value="TOU">Time-of-Use (TOU)</option>
            </select>
        </div>
    `;
}

function renderFixedChargeField() {
    const container = document.getElementById('tariff-controls');
    if (!container) return;

    const fixedChargeDiv = document.createElement('div');
    fixedChargeDiv.className = 'row g-3 mb-4';
    fixedChargeDiv.innerHTML = `
        <div class="col-md-6">
            <label for="monthly-fixed-charge" class="form-label fw-semibold">
                Monthly Fixed Charge ($)
            </label>
            <input type="number" id="monthly-fixed-charge" value="10.00" step="0.01" min="0"
                   class="form-control">
        </div>
        <div class="col-md-6">
            <label for="feed-in-tariff" class="form-label fw-semibold">
                Feed-in Tariff ($/kWh)
            </label>
            <input type="number" id="feed-in-tariff" value="0.05" step="0.0001" min="0"
                   class="form-control">
        </div>
    `;
    container.appendChild(fixedChargeDiv);

    document.getElementById('monthly-fixed-charge').addEventListener('input', updatePYSAMFields);
    document.getElementById('feed-in-tariff').addEventListener('input', updatePYSAMFields);
}

function setupTariffEventListeners() {
    const tariffSelect = document.getElementById('tariff-type-select');
    if (tariffSelect) {
        tariffSelect.addEventListener('change', (e) => {
            currentTariffType = e.target.value;
            updateTariffUI();
        });
    }
}

function updateTariffUI() {
    const detailsContainer = document.getElementById('tariff-config-details');
    if (!detailsContainer) return;

    detailsContainer.innerHTML = '';

    switch(currentTariffType) {
        case 'Flat':
            renderFlatRateUI();
            break;
        case 'TOU':
            renderTOUUI();
            break;
    }

    updatePYSAMFields();
}

function renderFlatRateUI() {
    const container = document.getElementById('tariff-config-details');
    if (!container) return;

    container.innerHTML = `
        <div class="rate-input-group">
            <h5 class="mb-3"><i class="fas fa-dollar-sign me-2"></i>Flat Rate Configuration</h5>
            <div class="row g-3">
                <div class="col-md-6">
                    <label for="flat-rate" class="form-label fw-semibold">
                        Energy Rate ($/kWh)
                    </label>
                    <input type="number" id="flat-rate" value="0.12" step="0.0001" min="0"
                           class="form-control">
                </div>
            </div>
        </div>
    `;

    const flatRateInput = document.getElementById('flat-rate');
    if (flatRateInput) {
        flatRateInput.addEventListener('input', updatePYSAMFields);
    }
}

function renderTOUUI() {
    const container = document.getElementById('tariff-config-details');
    const periodNames = getPeriodNames(currentTOUPeriodCount);

    container.innerHTML = `
        <div class="rate-input-group mb-4">
            <h5 class="mb-3"><i class="fas fa-clock me-2"></i>TOU Period Configuration</h5>
            <div class="row g-3">
                <div class="col-md-6">
                    <label for="tou-period-count" class="form-label fw-semibold">
                        Number of TOU Periods
                    </label>
                    <select id="tou-period-count" class="form-control">
                        <option value="2">2 (Peak/Off-Peak)</option>
                        <option value="3">3 (Peak/Shoulder/Off-Peak)</option>
                        <option value="4">4 (Peak/Mid-Peak/Shoulder/Off-Peak)</option>
                    </select>
                </div>
            </div>
        </div>

        <div id="tou-schedule-container" class="tou-schedule-grid mb-4">
            <h5 class="mb-3"><i class="fas fa-calendar-alt me-2"></i>Time Periods Configuration</h5>
            <div class="row g-3">
                <div class="col-md-6">
                    <div class="bg-light p-3 rounded">
                        <h6 class="fw-semibold mb-3">Weekday Schedule</h6>
                        ${generateScheduleGrid('weekday', periodNames)}
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="bg-light p-3 rounded">
                        <h6 class="fw-semibold mb-3">Weekend Schedule</h6>
                        ${generateScheduleGrid('weekend', periodNames)}
                    </div>
                </div>
            </div>
        </div>

        <div id="tou-rates-container" class="rate-input-group mb-4">
            <h5 class="mb-3"><i class="fas fa-chart-line me-2"></i>Energy Rates ($/kWh)</h5>
            <div class="row g-3">
                ${periodNames.map((name, index) => `
                    <div class="col-md-${12/currentTOUPeriodCount}">
                        <label class="form-label fw-semibold">${name} Rate</label>
                        <input type="number" class="tou-rate-input form-control"
                               data-period="${index + 1}" value="${index === 0 ? 0.25 : 0.12}"
                               step="0.0001" min="0">
                    </div>
                `).join('')}
            </div>
        </div>

        <div class="demand-charge-section">
            <div class="d-flex align-items-center justify-content-between mb-3">
                <h5 class="mb-0"><i class="fas fa-bolt me-2"></i>Demand Charges</h5>
                <div class="form-check form-switch">
                    <input class="form-check-input" type="checkbox" id="enable-demand-charges">
                    <label class="form-check-label fw-semibold" for="enable-demand-charges">
                        Enable Demand Charges
                    </label>
                </div>
            </div>

            <div id="demand-charge-fields" class="d-none">
                <div class="row g-3">
                    ${periodNames.map((name, index) => `
                        <div class="col-md-${12/currentTOUPeriodCount}">
                            <label class="form-label fw-semibold">
                                ${name} Demand ($/kW)
                            </label>
                            <input type="number" class="tou-demand-input form-control"
                                   data-period="${index + 1}" value="${index === 0 ? 15.00 : 0.00}"
                                   step="0.01" min="0">
                        </div>
                    `).join('')}
                </div>
                <div class="mt-2">
                    <small class="text-muted">
                        <strong>Australian Implementation:</strong> Charges apply to the maximum demand
                        within each TOU period. Typically only Peak periods carry demand charges.
                    </small>
                </div>
            </div>
        </div>
    `;

    document.getElementById('tou-period-count').addEventListener('change', (e) => {
        currentTOUPeriodCount = parseInt(e.target.value);
        renderTOUUI();
    });

    document.getElementById('enable-demand-charges').addEventListener('change', (e) => {
        const fieldsContainer = document.getElementById('demand-charge-fields');
        if (fieldsContainer) {
            fieldsContainer.classList.toggle('d-none', !e.target.checked);
        }
        updatePYSAMFields();
    });

    document.getElementById('tou-period-count').value = currentTOUPeriodCount;

    document.querySelectorAll('.tou-time-input').forEach(input => {
        input.addEventListener('change', updateTOUSchedule);
    });

    document.querySelectorAll('.tou-rate-input').forEach(input => {
        input.addEventListener('input', (e) => {
            const period = parseInt(e.target.dataset.period);
            touRates[period - 1] = parseFloat(e.target.value) || 0;
            updatePYSAMFields();
        });
    });

    document.querySelectorAll('.tou-demand-input').forEach(input => {
        input.addEventListener('input', updatePYSAMFields);
    });
}

function generateScheduleGrid(dayType, periodNames) {
    let html = '';

    periodNames.forEach((name, index) => {
        const periodNum = index + 1;

        html += `
            <div class="mb-3">
                <label class="form-label fw-semibold small">${name} Period (${periodNum})</label>
                <div class="row g-2">
                    <div class="col-6">
                        <label class="form-label small text-muted">Start Hour</label>
                        <select class="tou-time-input form-control form-control-sm"
                                data-day-type="${dayType}" data-period="${periodNum}" data-type="start">
                            ${generateHourOptions()}
                        </select>
                    </div>
                    <div class="col-6">
                        <label class="form-label small text-muted">End Hour</label>
                        <select class="tou-time-input form-control form-control-sm"
                                data-day-type="${dayType}" data-period="${periodNum}" data-type="end">
                            ${generateHourOptions()}
                        </select>
                    </div>
                </div>
            </div>
        `;
    });

    return html;
}

function updateTOUSchedule() {
    touWeekdaySchedule = Array(24).fill(1);
    touWeekendSchedule = Array(24).fill(1);

    for (let period = 1; period <= currentTOUPeriodCount; period++) {
        const startInput = document.querySelector(`.tou-time-input[data-day-type="weekday"][data-period="${period}"][data-type="start"]`);
        const endInput = document.querySelector(`.tou-time-input[data-day-type="weekday"][data-period="${period}"][data-type="end"]`);

        if (startInput && endInput) {
            const startHour = parseInt(startInput.value);
            const endHour = parseInt(endInput.value);

            if (startHour < endHour) {
                for (let h = startHour; h < endHour; h++) {
                    touWeekdaySchedule[h] = period;
                }
            } else {
                for (let h = startHour; h < 24; h++) {
                    touWeekdaySchedule[h] = period;
                }
                for (let h = 0; h < endHour; h++) {
                    touWeekdaySchedule[h] = period;
                }
            }
        }
    }

    for (let period = 1; period <= currentTOUPeriodCount; period++) {
        const startInput = document.querySelector(`.tou-time-input[data-day-type="weekend"][data-period="${period}"][data-type="start"]`);
        const endInput = document.querySelector(`.tou-time-input[data-day-type="weekend"][data-period="${period}"][data-type="end"]`);

        if (startInput && endInput) {
            const startHour = parseInt(startInput.value);
            const endHour = parseInt(endInput.value);

            if (startHour < endHour) {
                for (let h = startHour; h < endHour; h++) {
                    touWeekendSchedule[h] = period;
                }
            } else {
                for (let h = startHour; h < 24; h++) {
                    touWeekendSchedule[h] = period;
                }
                for (let h = 0; h < endHour; h++) {
                    touWeekendSchedule[h] = period;
                }
            }
        }
    }

    updatePYSAMFields();
}

function updatePYSAMFields() {
    let weekdayField = document.getElementById('ur_ec_sched_weekday');
    let weekendField = document.getElementById('ur_ec_sched_weekend');
    let touMatField = document.getElementById('ur_ec_tou_mat');
    let demandFlatField = document.getElementById('ur_dc_flat_mat');
    let fixedChargeField = document.getElementById('monthly_fixed_charge');
    let exportMatField = document.getElementById('ur_ec_export_mat');

    if (!exportMatField) {
        exportMatField = document.createElement('input');
        exportMatField.type = 'hidden';
        exportMatField.id = 'ur_ec_export_mat';
        exportMatField.dataset.pysamModule = 'UtilityRate5';
        exportMatField.dataset.pysamKey = 'ur_ec_export_mat';
        document.getElementById('tariff-config-container').appendChild(exportMatField);
    }

    if (!fixedChargeField) {
        fixedChargeField = document.createElement('input');
        fixedChargeField.type = 'hidden';
        fixedChargeField.id = 'monthly_fixed_charge';
        fixedChargeField.dataset.pysamModule = 'UtilityRate5';
        fixedChargeField.dataset.pysamKey = 'monthly_fixed_charge';
        document.getElementById('tariff-config-container').appendChild(fixedChargeField);
    }

    const fixedChargeInput = document.getElementById('monthly-fixed-charge');
    if (fixedChargeInput) {
        fixedChargeField.value = parseFloat(fixedChargeInput.value) || 10.00;
    }

    const feedInRate = parseFloat(document.getElementById('feed-in-tariff')?.value) || 0.05;
    exportMatField.value = JSON.stringify([
        [1, 1, 9.9999999999999998e+37, feedInRate]
    ]);

    if (currentTariffType === 'Flat') {
        const flatRate = parseFloat(document.getElementById('flat-rate')?.value) || 0.25;

        const monthlyWeekdaySchedule = Array(12).fill(touWeekdaySchedule);
        const monthlyWeekendSchedule = Array(12).fill(touWeekendSchedule);
        if (weekdayField) weekdayField.value = JSON.stringify(monthlyWeekdaySchedule);
        if (weekendField) weekendField.value = JSON.stringify(monthlyWeekendSchedule);

        if (touMatField) {
            touMatField.value = JSON.stringify([
                [1, 1, 9.9999999999999998e+37, feedInRate, flatRate, flatRate]
            ]);
        }
    }
    else if (currentTariffType === 'TOU') {
        const monthlyWeekdaySchedule = Array(12).fill(touWeekdaySchedule);
        const monthlyWeekendSchedule = Array(12).fill(touWeekendSchedule);
        if (weekdayField) weekdayField.value = JSON.stringify(monthlyWeekdaySchedule);
        if (weekendField) weekendField.value = JSON.stringify(monthlyWeekendSchedule);

        const currentRates = [];
        for (let i = 1; i <= currentTOUPeriodCount; i++) {
            const rateInput = document.querySelector(`.tou-rate-input[data-period="${i}"]`);
            if (rateInput) {
                currentRates[i-1] = parseFloat(rateInput.value) || 0;
            } else {
                currentRates[i-1] = touRates[i-1] || 0;
            }
        }

        const touMatrix = [];
        for (let i = 1; i <= currentTOUPeriodCount; i++) {
            touMatrix.push([
                i,
                1,
                9.9999999999999998e+37,
                feedInRate,
                currentRates[i-1],
                0
            ]);
        }
        if (touMatField) touMatField.value = JSON.stringify(touMatrix);

        console.log('TOU Rates collected:', currentRates);
        console.log('TOU Matrix:', touMatrix);
    }

    if (currentTariffType === 'TOU' && document.getElementById('enable-demand-charges')?.checked) {
        const demandMatrix = [];
        document.querySelectorAll('.tou-demand-input').forEach(input => {
            const period = parseInt(input.dataset.period);
            const rate = parseFloat(input.value) || 0;
            if (rate > 0) {
                demandMatrix.push([1, period, 9.999999e+37, rate]);
            }
        });
        if (demandFlatField) {
            demandFlatField.value = demandMatrix.length > 0 ? JSON.stringify(demandMatrix) : '';
        }
    } else {
        if (demandFlatField) demandFlatField.value = '';
    }
}

function getPeriodNames(count) {
    switch(count) {
        case 2: return ['Peak', 'Off-Peak'];
        case 3: return ['Peak', 'Shoulder', 'Off-Peak'];
        case 4: return ['Peak', 'Mid-Peak', 'Shoulder', 'Off-Peak'];
        default: return Array(count).fill('Period');
    }
}

function generateHourOptions() {
    let options = '';
    for (let i = 0; i < 24; i++) {
        const hour = i % 12 || 12;
        const ampm = i < 12 ? 'AM' : 'PM';
        options += `<option value="${i}">${hour}:00 ${ampm}</option>`;
    }
    return options;
}

// ========== ADDRESS FUNCTIONALITY ==========

function createAddressSection() {
    return `
        <div class="address-section">
            <div class="subsection-header">
                <h4><i class="fas fa-map-marker-alt me-2"></i>Project Location</h4>
            </div>
            <div class="row g-3">
                <div class="col-md-8">
                    <label for="project-address" class="form-label">Project Address</label>
                    <input type="text" class="form-control" id="project-address"
                           placeholder="Enter full project address">
                </div>
                <div class="col-md-4">
                    <label class="form-label">&nbsp;</label>
                    <button type="button" class="btn btn-primary w-100" id="geocode-btn">
                        <i class="fas fa-search-location me-2"></i>Locate
                    </button>
                </div>
                <div class="col-12">
                    <div id="map"></div>
                </div>
                <div class="col-md-6">
                    <label for="latitude" class="form-label">Latitude</label>
                    <input type="number" class="form-control" id="latitude"
                           data-pysam-module="site" data-pysam-key="lat"
                           min="-90" max="90" step="any" required>
                </div>
                <div class="col-md-6">
                    <label for="longitude" class="form-label">Longitude</label>
                    <input type="number" class="form-control" id="longitude"
                           data-pysam-module="site" data-pysam-key="lon"
                           min="-180" max="180" step="any" required>
                </div>
            </div>
        </div>
    `;
}

function initializeAddressSection() {
    if (typeof L !== 'undefined') {
        const mapElement = document.getElementById('map');
        if (mapElement && !mapElement._leaflet_id) {
            try {
                const map = L.map('map').setView([-25.2744, 133.7751], 5);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '&copy; OpenStreetMap contributors'
                }).addTo(map);

                console.log("Map initialized successfully");

                const geocodeBtn = document.getElementById('geocode-btn');
                if (geocodeBtn) {
                    geocodeBtn.addEventListener('click', function() {
                        const address = document.getElementById('project-address').value;
                        if (!address) {
                            showNotification('Please enter an address first', 'warning');
                            return;
                        }

                        this.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Locating...';
                        this.disabled = true;

                        const url = `/api/geocode?address=${encodeURIComponent(address)}`;

                        fetch(url)
                            .then(res => res.json())
                            .then(data => {
                                if (data.length) {
                                    const lat = parseFloat(data[0].lat);
                                    const lon = parseFloat(data[0].lon);
                                    map.setView([lat, lon], 13);
                                    L.marker([lat, lon]).addTo(map);
                                    document.getElementById('latitude').value = lat;
                                    document.getElementById('longitude').value = lon;
                                    showNotification('Location found successfully!', 'success');

                                    document.querySelectorAll('[data-pysam-key="lat"], [data-pysam-key="latitude"]').forEach(el => {
                                        el.value = lat;
                                        el.setAttribute('value', lat);
                                    });
                                    document.querySelectorAll('[data-pysam-key="lon"], [data-pysam-key="longitude"]').forEach(el => {
                                        el.value = lon;
                                        el.setAttribute('value', lon);
                                    });

                                    const latEvent = new Event('change', { bubbles: true });
                                    const lonEvent = new Event('change', { bubbles: true });
                                    document.getElementById('latitude').dispatchEvent(latEvent);
                                    document.getElementById('longitude').dispatchEvent(lonEvent);

                                    setTimeout(() => {
                                        const detectedRegion = detectFCASRegion(lat, lon);
                                        fcasState.region = detectedRegion;
                                        const fcasRegionSelect = document.getElementById('fcas-region');
                                        if (fcasRegionSelect) {
                                            fcasRegionSelect.value = detectedRegion;
                                            updateFCASRevenue();
                                            updateFCASHiddenFields();
                                        }
                                    }, 100);
                                } else {
                                    showNotification('Address not found. Please try a different address.', 'warning');
                                }
                            })
                            .catch(err => {
                                console.error('Geocoding error:', err);
                                showNotification('Error locating address. Please try again.', 'error');
                            })
                            .finally(() => {
                                this.innerHTML = '<i class="fas fa-search-location me-2"></i>Locate';
                                this.disabled = false;
                            });
                    });
                }
            } catch (error) {
                console.error('Map initialization error:', error);
                mapElement.innerHTML = '<div class="p-3 text-center text-muted">Map initialization failed. Please enter coordinates manually.</div>';
            }
        }
    } else {
        console.log("Leaflet not available, map disabled");
        const mapElement = document.getElementById('map');
        if (mapElement) {
            mapElement.innerHTML = '<div class="p-3 text-center text-muted">Map requires Leaflet library. Please enter coordinates manually.</div>';
        }
    }
}

function createBatteryDispatchField(field) {
    const fieldId = field.key;

    const dispatchOptions = [
        {
            value: "0",
            name: "Peak Shaving",
            description: "Reduces peak demand charges while providing FCAS services",
            icon: "📊"
        },
        {
            value: "4",
            name: "Smart Energy Trading (Recommended)",
            description: "Optimizes time-of-use arbitrage + FCAS revenue automatically",
            icon: "⚡",
            recommended: true
        },
        {
            value: "5",
            name: "Self Consumption",
            description: "Maximizes on-site consumption while providing FCAS services",
            icon: "🏠"
        }
    ];

    let fieldHTML = '<div class="mb-3">';

    fieldHTML += `<label for="${fieldId}" class="form-label fw-semibold">`;
    fieldHTML += `${field.label}`;
    if (field.required) fieldHTML += '<span class="text-danger">*</span>';
    fieldHTML += '</label>';

    fieldHTML += `<select class="form-control" id="${fieldId}" data-pysam-key="${field.key}"`;
    if (field.module) fieldHTML += ` data-pysam-module="${field.module}"`;
    if (field.required) fieldHTML += ' required';
    fieldHTML += ' onchange="updateBatteryDispatchDescription(this.value)">';

    dispatchOptions.forEach(option => {
        const selected = option.recommended ? 'selected' : '';
        const recommendedText = option.recommended ? ' (Recommended)' : '';
        fieldHTML += `<option value="${option.value}" ${selected}>`;
        fieldHTML += `${option.icon} ${option.name}${recommendedText}`;
        fieldHTML += '</option>';
    });

    fieldHTML += '</select>';

    fieldHTML += `
        <div class="mt-3 p-3 bg-light rounded" id="battery-dispatch-description">
            <div class="d-flex align-items-start">
                <div class="me-3">
                    <span class="fs-4" id="dispatch-icon">⚡</span>
                </div>
                <div class="flex-grow-1">
                    <h6 class="mb-1" id="dispatch-name">Smart Energy Trading (Recommended)</h6>
                    <p class="mb-0 text-muted small" id="dispatch-desc">
                        Optimizes time-of-use arbitrage + FCAS revenue automatically
                    </p>
                </div>
            </div>
        </div>
    `;

    fieldHTML += '</div>';
    return fieldHTML;
}

window.updateBatteryDispatchDescription = function(selectedValue) {
    const dispatchOptions = [
        {
            value: "0",
            name: "Peak Shaving",
            description: "Reduces peak demand charges while providing FCAS services",
            icon: "📊"
        },
        {
            value: "4",
            name: "Smart Energy Trading",
            description: "Optimizes time-of-use arbitrage + FCAS revenue automatically",
            icon: "⚡",
            recommended: true
        },
        {
            value: "5",
            name: "Self Consumption",
            description: "Maximizes on-site consumption while providing FCAS services",
            icon: "🏠"
        }
    ];

    const option = dispatchOptions.find(opt => opt.value === selectedValue);
    if (!option) return;

    const iconEl = document.getElementById('dispatch-icon');
    const nameEl = document.getElementById('dispatch-name');
    const descEl = document.getElementById('dispatch-desc');

    if (iconEl) iconEl.textContent = option.icon;
    if (nameEl) nameEl.textContent = option.name + (option.recommended ? ' (Recommended)' : '');
    if (descEl) descEl.textContent = option.description;
};

function ensureBatteryDispatchConsistency() {
    const batteryEnabled = document.getElementById('input__en_batt')?.value === 'Yes';
    if (!batteryEnabled) return;

    const dispatchSelect = document.getElementById('input__batt_dispatch_choice');
    if (dispatchSelect && !dispatchSelect.value) {
        dispatchSelect.value = "4";
        updateBatteryDispatchDescription("4");
        console.log("Set default battery dispatch to Smart Energy Trading (4)");
    }
}

// ========== FORM CREATION AND MANAGEMENT ==========

function getInputType(typeString) {
    if (!typeString) return 'text';
    const lower = typeString.toLowerCase();
    if (lower.includes('dropdown') || lower.includes('select')) return 'select';
    if (lower.includes('checkbox') || lower.includes('boolean')) return 'checkbox';
    if (lower.includes('date')) return 'date';
    if (lower.includes('percentage')) return 'percentage';
    if (lower.includes('currency')) return 'currency';
    if (lower.includes('address')) return 'address';
    if (lower.includes('tou_table') || lower.includes('demand_table')) return 'special_table';
    if (lower.includes('number') || lower.includes('float') || lower.includes('int')) return 'number';
    return 'text';
}

function showNotification(message, type) {
    type = type || 'info';
    const notification = document.createElement('div');
    notification.className = 'alert alert-' + (type === 'error' ? 'danger' : type) + ' alert-dismissible fade show position-fixed';
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = message + '<button type="button" class="btn-close" data-bs-dismiss="alert"></button>';

    document.body.appendChild(notification);

    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

function createFormField(field) {
    if (!field || !field.label) return '';

    if (field.key === 'batt_dispatch_choice') {
        return createBatteryDispatchField(field);
    }

    const fieldType = getInputType(field.type);
    const fieldId = field.key;

    if (fieldType === 'address') return '';

    if (field.special_handling === 'fcas_config' || field.module === 'Enhanced_FCAS') return '';

    let fieldHTML = '<div class="mb-3">';

    if (fieldType !== 'checkbox') {
        fieldHTML += '<label for="' + fieldId + '" class="form-label">';
        fieldHTML += field.label;
        if (field.required) {
            fieldHTML += '<span class="text-danger">*</span>';
        }
        fieldHTML += '</label>';
    }

    if (fieldType === 'select') {
        fieldHTML += '<select class="form-control" id="' + fieldId + '" data-pysam-key="' + field.key + '"';
        if (field.module) fieldHTML += ' data-pysam-module="' + field.module + '"';
        if (field.required) fieldHTML += ' required';
        fieldHTML += '>';
        fieldHTML += '<option value="">-- Select --</option>';

        if (field.description && field.description.includes(';')) {
            const options = field.description.split(';');
            for (let i = 0; i < options.length; i++) {
                const opt = options[i].trim();
                if (opt.includes(':')) {
                    const parts = opt.split(':');
                    const val = parts[0].trim();
                    const text = parts[1].trim();
                    fieldHTML += '<option value="' + val + '">' + text + '</option>';
                } else if (opt) {
                    fieldHTML += '<option value="' + opt + '">' + opt + '</option>';
                }
            }
        }
        fieldHTML += '</select>';

    } else if (fieldType === 'checkbox') {
        fieldHTML += '<div class="form-check">';
        fieldHTML += '<input type="checkbox" class="form-check-input" id="' + fieldId + '" data-pysam-key="' + field.key + '"';
        if (field.module) fieldHTML += ' data-pysam-module="' + field.module + '"';
        fieldHTML += '>';
        fieldHTML += '<label class="form-check-label" for="' + fieldId + '">' + field.label + '</label>';
        fieldHTML += '</div>';

    } else if (fieldType === 'percentage') {
        fieldHTML += '<div class="input-group">';
        fieldHTML += '<input type="number" class="form-control" id="' + fieldId + '" data-pysam-key="' + field.key + '"';
        if (field.module) fieldHTML += ' data-pysam-module="' + field.module + '"';
        fieldHTML += ' step="0.1" min="0" max="100"';
        if (field.default_value) fieldHTML += ' value="' + field.default_value + '"';
        if (field.required) fieldHTML += ' required';
        fieldHTML += '>';
        fieldHTML += '<span class="input-group-text">%</span>';
        fieldHTML += '</div>';

    } else if (fieldType === 'currency') {
        fieldHTML += '<div class="input-group">';
        fieldHTML += '<span class="input-group-text">$</span>';
        fieldHTML += '<input type="number" class="form-control" id="' + fieldId + '" data-pysam-key="' + field.key + '"';
        if (field.module) fieldHTML += ' data-pysam-module="' + field.module + '"';
        fieldHTML += ' step="0.01" min="0"';
        if (field.default_value) fieldHTML += ' value="' + field.default_value + '"';
        if (field.required) fieldHTML += ' required';
        fieldHTML += '>';
        fieldHTML += '</div>';

    } else if (fieldType === 'special_table') {
        fieldHTML += '<div class="d-flex align-items-center">';
        fieldHTML += '<input type="hidden" id="' + field.key + '" data-pysam-key="' + field.key + '"';
        if (field.module) fieldHTML += ' data-pysam-module="' + field.module + '"';
        fieldHTML += '>';
        fieldHTML += '<button type="button" class="btn btn-outline-primary" onclick="alert(\'Table editor not yet implemented\')">';
        fieldHTML += '<i class="fas fa-table me-2"></i>Configure ' + field.label;
        fieldHTML += '</button>';
        fieldHTML += '</div>';

    } else {
        const inputType = fieldType === 'number' ? 'number' : fieldType === 'date' ? 'date' : 'text';
        fieldHTML += '<input type="' + inputType + '" class="form-control" id="' + fieldId + '" data-pysam-key="' + field.key + '"';
        if (field.module) fieldHTML += ' data-pysam-module="' + field.module + '"';
        if (fieldType === 'number' && field.type && field.type.includes('float')) {
            fieldHTML += ' step="0.01"';
        }
        if (field.validation && field.validation.min !== undefined) {
            fieldHTML += ' min="' + field.validation.min + '"';
        }
        if (field.validation && field.validation.max !== undefined) {
            fieldHTML += ' max="' + field.validation.max + '"';
        }
        if (field.default_value) fieldHTML += ' value="' + field.default_value + '"';
        if (field.required) fieldHTML += ' required';
        fieldHTML += '>';
    }

    if (field.description && !field.description.includes(';')) {
        fieldHTML += '<div class="form-text text-muted small">' + field.description + '</div>';
    }

    fieldHTML += '</div>';
    return fieldHTML;
}

function createEnhancedForm(config) {
    const root = document.getElementById('dynamic-form-root');
    if (!root) {
        console.error("No form root found");
        return;
    }

    const sections = [...new Set(config.map(field => field.section).filter(Boolean))]
        .sort((a, b) => {
            const orderA = TAB_CONFIG[a] ? TAB_CONFIG[a].order : 999;
            const orderB = TAB_CONFIG[b] ? TAB_CONFIG[b].order : 999;
            return orderA - orderB;
        });

    console.log("Sections found:", sections);

    let tabsHTML = '<ul class="nav nav-tabs nav-tabs-custom mb-4">';
    sections.forEach((section, index) => {
        const isActive = index === 0 ? 'active' : '';
        const tabId = section.toLowerCase().replace(/\s+/g, '-');
        const config = TAB_CONFIG[section] || { icon: 'fas fa-cog' };

        tabsHTML += '<li class="nav-item">';
        tabsHTML += '<a class="nav-link ' + isActive + '" href="#' + tabId + '" data-tab="' + tabId + '">';
        tabsHTML += '<i class="' + config.icon + ' me-2"></i>';
        tabsHTML += section;
        tabsHTML += '</a>';
        tabsHTML += '</li>';
    });
    tabsHTML += '</ul>';

    let contentHTML = '<div class="tab-content">';
    sections.forEach((section, index) => {
        const isActive = index === 0 ? 'show active' : '';
        const tabId = section.toLowerCase().replace(/\s+/g, '-');
        const sectionFields = config.filter(field => field.section === section);

        contentHTML += '<div class="tab-pane fade ' + isActive + '" id="' + tabId + '">';
        contentHTML += '<div class="form-card p-4">';
        contentHTML += '<h3 class="mb-4">';
        contentHTML += '<i class="' + (TAB_CONFIG[section] ? TAB_CONFIG[section].icon : 'fas fa-cog') + ' me-2 text-primary"></i>';
        contentHTML += section;
        contentHTML += '</h3>';

        if (section === 'Project Information') {
            contentHTML += createAddressSection();

            const nonAddressFields = sectionFields.filter(field => getInputType(field.type) !== 'address');
            if (nonAddressFields.length > 0) {
                contentHTML += '<div class="subsection-header mt-4">';
                contentHTML += '<h4><i class="fas fa-info-circle me-2"></i>Project Details</h4>';
                contentHTML += '</div>';
                contentHTML += '<div class="row">';
                nonAddressFields.forEach(field => {
                    if (field.label && field.key) {
                        contentHTML += '<div class="col-md-6">';
                        contentHTML += createFormField(field);
                        contentHTML += '</div>';
                    }
                });
                contentHTML += '</div>';
            }
        }
        else if (section === 'Network Configuration') {
            contentHTML += createTariffSection();

            const nonTariffFields = sectionFields.filter(field => !field.special_handling || field.special_handling !== 'tariff');
            if (nonTariffFields.length > 0) {
                contentHTML += '<div class="subsection-header mt-4">';
                contentHTML += '<h4><i class="fas fa-network-wired me-2"></i>Other Network Settings</h4>';
                contentHTML += '</div>';
                contentHTML += '<div class="row">';
                nonTariffFields.forEach(field => {
                    if (field.label && field.key) {
                        contentHTML += '<div class="col-md-6">';
                        contentHTML += createFormField(field);
                        contentHTML += '</div>';
                    }
                });
                contentHTML += '</div>';
            }
        }
        else if (section === 'Financial Parameters') {
            const subsections = {};
            sectionFields.forEach(field => {
                const subsection = field.subsection || 'General';
                if (!subsections[subsection]) subsections[subsection] = [];
                subsections[subsection].push(field);
            });

            if (subsections['General']) {
                contentHTML += '<div class="row">';
                subsections['General'].forEach(field => {
                    if (field.label && field.key && field.module !== 'Enhanced_FCAS') {
                        contentHTML += '<div class="col-md-6">';
                        contentHTML += createFormField(field);
                        contentHTML += '</div>';
                    }
                });
                contentHTML += '</div>';
            }

            if (subsections['FCAS Markets']) {
                contentHTML += createFCASSection();
            }

            Object.entries(subsections).forEach(([subsectionName, subsectionFields]) => {
                if (subsectionName !== 'General' && subsectionName !== 'FCAS Markets') {
                    contentHTML += '<div class="subsection-header">';
                    contentHTML += '<h4><i class="fas fa-layer-group me-2"></i>' + subsectionName + '</h4>';
                    contentHTML += '</div>';
                    contentHTML += '<div class="row">';
                    subsectionFields.forEach(field => {
                        if (field.label && field.key) {
                            contentHTML += '<div class="col-md-6">';
                            contentHTML += createFormField(field);
                            contentHTML += '</div>';
                        }
                    });
                    contentHTML += '</div>';
                }
            });
        } else if (section === 'Load') {
            contentHTML += createLoadSection();
        } else {
            const subsections = {};
            sectionFields.forEach(field => {
                const subsection = field.subsection || 'General';
                if (!subsections[subsection]) subsections[subsection] = [];
                subsections[subsection].push(field);
            });

            Object.entries(subsections).forEach(([subsectionName, subsectionFields]) => {
                if (subsectionName !== 'General') {
                    contentHTML += '<div class="subsection-header">';
                    contentHTML += '<h4><i class="fas fa-layer-group me-2"></i>' + subsectionName + '</h4>';
                    contentHTML += '</div>';
                }

                contentHTML += '<div class="row">';
                subsectionFields.forEach(field => {
                    if (field.label && field.key) {
                        contentHTML += '<div class="col-md-6">';
                        contentHTML += createFormField(field);
                        contentHTML += '</div>';
                    }
                });
                contentHTML += '</div>';
            });
        }

        contentHTML += '</div>';
        contentHTML += '</div>';
    });
    contentHTML += '</div>';

    root.innerHTML = tabsHTML + contentHTML;

    const tabLinks = root.querySelectorAll('[data-tab]');
    tabLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();

            root.querySelectorAll('.nav-link').forEach(tab => tab.classList.remove('active'));
            root.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('show', 'active'));

            this.classList.add('active');
            const targetId = this.getAttribute('data-tab');
            const targetPane = document.getElementById(targetId);
            if (targetPane) {
                targetPane.classList.add('show', 'active');
            }

            updateProgressIndicator();

            if (targetId === 'project-information' && !addressSectionInitialized) {
                setTimeout(() => {
                    initializeAddressSection();
                    addressSectionInitialized = true;
                }, 100);
            }

            if (targetId === 'network-configuration' && !tariffSectionInitialized) {
                setTimeout(() => {
                    initTariffSection();
                    tariffSectionInitialized = true;
                }, 100);
            }

            if (targetId === 'financial-parameters' && !fcasSectionInitialized) {
                setTimeout(() => {
                    initFCASSection();
                    fcasSectionInitialized = true;
                }, 100);
            }

            if (targetId === 'load' && !loadSectionInitialized) {
                setTimeout(() => {
                    initLoadSection();
                    loadSectionInitialized = true;
                }, 100);
            }

            console.log('Tab switched to:', targetId);
        });
    });

    console.log("Enhanced form with address, tariff, FCAS, and load sections created");
}

function updateProgressIndicator() {
    const totalTabs = document.querySelectorAll('.nav-link').length;
    const activeTabIndex = Array.from(document.querySelectorAll('.nav-link')).findIndex(tab => tab.classList.contains('active'));
    const progress = ((activeTabIndex + 1) / totalTabs) * 100;

    const indicator = document.querySelector('.progress-indicator');
    if (indicator) {
        indicator.style.transform = 'scaleX(' + (progress / 100) + ')';
    }
}

async function initEnhancedForm() {
    try {
        console.log("Initializing complete enhanced form with FCAS and Load integration...");

        injectEnhancedCSS();

        const progressIndicator = document.createElement('div');
        progressIndicator.className = 'progress-indicator';
        document.body.appendChild(progressIndicator);

        const config = await loadFormConfig();

        if (config.length > 0) {
            createEnhancedForm(config);

            setTimeout(() => {
                config.forEach(field => {
                    if (field.default_value !== undefined) {
                        const input = document.getElementById('input__' + field.key);
                        if (input) {
                            if (input.type === 'checkbox') {
                                input.checked = field.default_value === 'true' || field.default_value === true;
                            } else {
                                input.value = field.default_value;
                            }
                        }
                    }
                });
            }, 100);

            setTimeout(() => initializeAddressSection(), 200);
            setTimeout(() => initTariffSection(), 300);
            setTimeout(() => initFCASSection(), 400);
            setTimeout(() => initLoadSection(), 500);

            updateProgressIndicator();
        } else {
            document.getElementById('dynamic-form-root').innerHTML = '<div class="alert alert-warning">No form configuration found</div>';
        }
    } catch (error) {
        console.error("Form initialization error:", error);
        document.getElementById('dynamic-form-root').innerHTML = '<div class="alert alert-danger">Error: ' + error.message + '</div>';
    }
}

function createFallbackNotification(message, type) {
    const notification = document.createElement('div');
    notification.className = 'alert alert-' + (type === 'error' ? 'danger' : type) + ' alert-dismissible fade show position-fixed';
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = message + '<button type="button" class="btn-close" data-bs-dismiss="alert"></button>';

    document.body.appendChild(notification);

    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

function serializeUpdatedHTML() {
    const rootEl = document.getElementById('dynamic-form-root');
    const tariffEl = document.getElementById('tariff-config-container');
    const fcasEl = document.getElementById('fcas-config-container');
    const loadEl = document.getElementById('load-config-container');

    if (!rootEl) return '';

    const rootClone = rootEl.cloneNode(true);
    const tariffClone = tariffEl ? tariffEl.cloneNode(true) : null;
    const fcasClone = fcasEl ? fcasEl.cloneNode(true) : null;
    const loadClone = loadEl ? loadEl.cloneNode(true) : null;

    const liveInputs = document.querySelectorAll('input, select, textarea');
    liveInputs.forEach(input => {
        const name = input.name || input.id || input.getAttribute('data-pysam-key');
        if (!name) return;

        const selectors = [
            `[name="${name}"]`,
            `[id="${name}"]`,
            `[data-pysam-key="${name}"]`
        ].join(',');

        let clonedInput = rootClone.querySelector(selectors);
        if (!clonedInput && tariffClone) {
            clonedInput = tariffClone.querySelector(selectors);
        }
        if (!clonedInput && fcasClone) {
            clonedInput = fcasClone.querySelector(selectors);
        }
        if (!clonedInput && loadClone) {
            clonedInput = loadClone.querySelector(selectors);
        }

        if (clonedInput) {
            if (input.type === 'checkbox' || input.type === 'radio') {
                clonedInput.checked = input.checked;
                if (input.checked) {
                    clonedInput.setAttribute('checked', 'checked');
                    clonedInput.removeAttribute('checked');
                }
            } else {
                clonedInput.value = input.value;
                clonedInput.setAttribute('value', input.value);
            }

            if (input.dataset.pysamKey) {
                clonedInput.setAttribute('data-pysam-key', input.dataset.pysamKey);
            }
            if (input.dataset.pysamModule) {
                clonedInput.setAttribute('data-pysam-module', input.dataset.pysamModule);
            }
        }
    });

    const wrapper = document.createElement('div');
    wrapper.appendChild(rootClone);
    if (tariffClone) {
        wrapper.appendChild(tariffClone);
    }
    if (fcasClone) {
        wrapper.appendChild(fcasClone);
    }
    if (loadClone) {
        wrapper.appendChild(loadClone);
    }

    return wrapper.innerHTML;
}

// Global function exports for main application
window.selectLoadMethod = selectLoadMethod;
window.updateAutoGeneration = updateAutoGeneration;
window.generateAutoLoad = generateAutoLoad;
window.updateMonthlyTotal = updateMonthlyTotal;
window.fillMonthlyAverage = fillMonthlyAverage;
window.clearMonthlyData = clearMonthlyData;
window.generateManualLoad = generateManualLoad;
window.handleDragOver = handleDragOver;
window.handleDragLeave = handleDragLeave;
window.handleFileDrop = handleFileDrop;
window.handleFileSelect = handleFileSelect;

// Start when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    initEnhancedForm().then(() => {
        setTimeout(() => {
            ensureBatteryDispatchConsistency();

            const batteryEnableField = document.getElementById('input__en_batt');
            if (batteryEnableField) {
                batteryEnableField.addEventListener('change', () => {
                    setTimeout(ensureBatteryDispatchConsistency, 100);
                });
            }
        }, 1000);
    });
});

console.log("Complete enhanced form script with FCAS and Load integration loaded");