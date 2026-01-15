// Main JavaScript for Multi-Disease Risk Prediction System

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resultsSection = document.getElementById('results');
    const loadingSection = document.getElementById('loading');
    const errorSection = document.getElementById('error');
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Hide previous results and errors
        resultsSection.style.display = 'none';
        errorSection.style.display = 'none';
        loadingSection.style.display = 'block';
        
        // Collect form data
        const formData = new FormData(form);
        const data = {};
        
        for (let [key, value] of formData.entries()) {
            data[key] = parseFloat(value);
        }
        
        try {
            // Send prediction request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            loadingSection.style.display = 'none';
            
            if (result.success) {
                displayResults(result);
            } else {
                showError(result.error || 'An error occurred during prediction');
            }
        } catch (error) {
            loadingSection.style.display = 'none';
            showError('Failed to connect to the server. Please try again.');
            console.error('Error:', error);
        }
    });
});

function displayResults(result) {
    const resultsSection = document.getElementById('results');
    const riskLevel = document.getElementById('riskLevel');
    const riskScore = document.getElementById('riskScore');
    const confidence = document.getElementById('confidence');
    const riskBar = document.getElementById('riskBar');
    
    // Display prediction
    riskLevel.textContent = result.prediction;
    riskLevel.className = result.prediction === 'High Risk' ? 'risk-level-high' : 'risk-level-low';
    
    riskScore.textContent = result.risk_score.toFixed(1);
    confidence.textContent = result.confidence.toFixed(1);
    
    // Update risk meter
    riskBar.style.width = result.risk_score + '%';
    
    // Display SHAP features
    const shapFeatures = document.getElementById('shapFeatures');
    shapFeatures.innerHTML = '';
    result.explanation.shap_features.forEach(([feature, impact]) => {
        const li = document.createElement('li');
        li.className = impact > 0 ? 'feature-positive' : 'feature-negative';
        li.innerHTML = `
            <span>${formatFeatureName(feature)}</span>
            <span style="font-weight: bold; color: ${impact > 0 ? '#e74c3c' : '#27ae60'}">
                ${impact > 0 ? '+' : ''}${impact.toFixed(4)}
            </span>
        `;
        shapFeatures.appendChild(li);
    });
    
    // Display LIME features
    const limeFeatures = document.getElementById('limeFeatures');
    limeFeatures.innerHTML = '';
    result.explanation.lime_features.forEach(([feature, weight]) => {
        const li = document.createElement('li');
        li.className = weight > 0 ? 'feature-positive' : 'feature-negative';
        li.innerHTML = `
            <span>${feature}</span>
            <span style="font-weight: bold; color: ${weight > 0 ? '#e74c3c' : '#27ae60'}">
                ${weight > 0 ? '+' : ''}${weight.toFixed(4)}
            </span>
        `;
        limeFeatures.appendChild(li);
    });
    
    // Display plots
    document.getElementById('shapPlot').src = 'data:image/png;base64,' + result.explanation.shap_plot;
    document.getElementById('limePlot').src = 'data:image/png;base64,' + result.explanation.lime_plot;
    
    // Show results
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showError(message) {
    const errorSection = document.getElementById('error');
    errorSection.textContent = message;
    errorSection.style.display = 'block';
}

function showTab(tabName) {
    // Hide all tab contents
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => {
        content.classList.remove('active');
    });
    
    // Remove active class from all buttons
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
}

function formatFeatureName(name) {
    // Convert snake_case to Title Case
    return name
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}
