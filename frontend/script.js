/**
 * Potato Disease Classification - Frontend JavaScript
 * Handles image upload, API calls, and result visualization
 */

// API endpoint
const API_URL = 'http://localhost:5000';

// Store uploaded image
let uploadedImage = null;

// Chart instances
let confidenceChart = null;
let probabilityChart = null;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    const imageInput = document.getElementById('imageInput');
    imageInput.addEventListener('change', handleImageUpload);
});

/**
 * Handle image file upload
 */
function handleImageUpload(event) {
    const file = event.target.files[0];
    
    if (!file) {
        return;
    }
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file');
        return;
    }
    
    uploadedImage = file;
    
    // Show image preview
    const reader = new FileReader();
    reader.onload = function(e) {
        const previewImg = document.getElementById('previewImg');
        previewImg.src = e.target.result;
        
        const imagePreview = document.getElementById('imagePreview');
        imagePreview.style.display = 'block';
        
        // Hide previous results
        document.getElementById('resultsSection').style.display = 'none';
    };
    reader.readAsDataURL(file);
}

/**
 * Compare models - Send image to backend
 */
async function compareModels() {
    if (!uploadedImage) {
        alert('Please upload an image first');
        return;
    }
    
    // Show loading spinner
    document.getElementById('loadingSpinner').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    
    // Prepare form data
    const formData = new FormData();
    formData.append('image', uploadedImage);
    
    try {
        // Send request to Flask backend
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const data = await response.json();
        
        // Hide loading spinner
        document.getElementById('loadingSpinner').style.display = 'none';
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('loadingSpinner').style.display = 'none';
        alert('Error connecting to backend. Make sure Flask server is running on port 5000.');
    }
}

/**
 * Display prediction results
 */
function displayResults(data) {
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    
    // === Custom CNN Results ===
    document.getElementById('cnnPrediction').textContent = data.custom_cnn.prediction;
    document.getElementById('cnnConfidence').textContent = `${data.custom_cnn.confidence}% Confidence`;
    document.getElementById('cnnTime').textContent = `${data.custom_cnn.inference_time_ms} ms`;
    
    const cnnBar = document.getElementById('cnnConfidenceBar');
    cnnBar.style.width = `${data.custom_cnn.confidence}%`;
    
    // === MobileNetV2 Results ===
    document.getElementById('mobilenetPrediction').textContent = data.mobilenet.prediction;
    document.getElementById('mobilenetConfidence').textContent = `${data.mobilenet.confidence}% Confidence`;
    document.getElementById('mobilenetTime').textContent = `${data.mobilenet.inference_time_ms} ms`;
    
    const mobileBar = document.getElementById('mobilenetConfidenceBar');
    mobileBar.style.width = `${data.mobilenet.confidence}%`;
    
    // === Agreement Indicator ===
    const agreementBox = document.getElementById('agreementBox');
    const agreementIcon = document.getElementById('agreementIcon');
    const agreementText = document.getElementById('agreementText');
    
    if (data.agreement) {
        agreementBox.className = 'agreement-box agree';
        agreementIcon.textContent = '✅';
        agreementText.textContent = 'Both models agree on the prediction!';
    } else {
        agreementBox.className = 'agreement-box disagree';
        agreementIcon.textContent = '⚠️';
        agreementText.textContent = 'Models have different predictions. Consider reviewing the image.';
    }
    
    // === Create Charts ===
    createConfidenceChart(data);
    createProbabilityChart(data);
}

/**
 * Create confidence comparison bar chart
 */
function createConfidenceChart(data) {
    const ctx = document.getElementById('confidenceChart').getContext('2d');
    
    // Destroy previous chart if exists
    if (confidenceChart) {
        confidenceChart.destroy();
    }
    
    confidenceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Custom CNN', 'MobileNetV2'],
            datasets: [{
                label: 'Confidence (%)',
                data: [data.custom_cnn.confidence, data.mobilenet.confidence],
                backgroundColor: [
                    'rgba(102, 126, 234, 0.8)',
                    'rgba(245, 87, 108, 0.8)'
                ],
                borderColor: [
                    'rgba(102, 126, 234, 1)',
                    'rgba(245, 87, 108, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y.toFixed(2) + '% confidence';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Create class probability distribution chart
 */
function createProbabilityChart(data) {
    const ctx = document.getElementById('probabilityChart').getContext('2d');
    
    // Destroy previous chart if exists
    if (probabilityChart) {
        probabilityChart.destroy();
    }
    
    const classNames = Object.keys(data.custom_cnn.class_probabilities);
    
    probabilityChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: classNames,
            datasets: [
                {
                    label: 'Custom CNN',
                    data: Object.values(data.custom_cnn.class_probabilities),
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 2
                },
                {
                    label: 'MobileNetV2',
                    data: Object.values(data.mobilenet.class_probabilities),
                    backgroundColor: 'rgba(245, 87, 108, 0.2)',
                    borderColor: 'rgba(245, 87, 108, 1)',
                    pointBackgroundColor: 'rgba(245, 87, 108, 1)',
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.r.toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
}