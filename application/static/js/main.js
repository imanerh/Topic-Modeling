// DOM Elements
const documentInput = document.getElementById('documentInput');
const classifyBtn = document.getElementById('classifyBtn');
const classifyBtnText = document.getElementById('classifyBtnText');
const classifySpinner = document.getElementById('classifySpinner');
const clearBtn = document.getElementById('clearBtn');
const showExamplesBtn = document.getElementById('showExamplesBtn');
const examplesContainer = document.getElementById('examplesContainer');
const exampleItems = document.querySelectorAll('.example-item');
const errorAlert = document.getElementById('errorAlert');

// Results Elements
const initialState = document.getElementById('initialState');
const loadingState = document.getElementById('loadingState');
const resultsState = document.getElementById('resultsState');
const topTopicLabel = document.getElementById('topTopicLabel');
const topTopicConfidence = document.getElementById('topTopicConfidence');
const confidenceScores = document.getElementById('confidenceScores');
const keyTerms = document.getElementById('keyTerms');

// Initialize chart
let topicChart = null;
let userDocumentPoint = null;

// Utility function to create confidence score bar
function createScoreBar(score) {
    const container = document.createElement('div');
    container.className = 'mb-2';
    
    const labelRow = document.createElement('div');
    labelRow.className = 'd-flex justify-content-between align-items-center mb-1';
    
    const labelText = document.createElement('div');
    labelText.className = 'd-flex align-items-center';
    
    // Get color with case-insensitive lookup or use the color directly from score
    const color = score.color || 
                 topicColors[score.label.toLowerCase()] || 
                 '#999';
    
    const colorDot = document.createElement('span');
    colorDot.className = 'me-2 rounded-circle';
    colorDot.style.width = '10px';
    colorDot.style.height = '10px';
    colorDot.style.display = 'inline-block';
    colorDot.style.backgroundColor = color;
    
    const label = document.createElement('span');
    label.textContent = score.label;
    label.className = 'small';
    
    const probability = document.createElement('span');
    probability.textContent = `${Math.round(score.probability * 100)}%`;
    probability.className = 'small text-muted';
    
    labelText.appendChild(colorDot);
    labelText.appendChild(label);
    labelRow.appendChild(labelText);
    labelRow.appendChild(probability);
    
    const progressContainer = document.createElement('div');
    progressContainer.className = 'progress';
    
    const progressBar = document.createElement('div');
    progressBar.className = 'progress-bar';
    progressBar.style.width = `${Math.round(score.probability * 100)}%`;
    progressBar.style.backgroundColor = color;
    
    progressContainer.appendChild(progressBar);
    
    container.appendChild(labelRow);
    container.appendChild(progressContainer);
    
    return container;
}

// Initialize the chart
function initChart() {
    const ctx = document.getElementById('topicChart').getContext('2d');
    
    // Prepare data
    const datasets = [];
    
    // Create a dataset for each unique topic label
    const uniqueLabels = [...new Set(topicCoordinates.map(t => t.label))];
    
    uniqueLabels.forEach(label => {
        // Get color using case-insensitive lookup
        const points = topicCoordinates.filter(t => t.label === label);
        let color;
        
        // First try to get color from the first point that has it
        const pointWithColor = points.find(p => p.color);
        if (pointWithColor && pointWithColor.color) {
            color = pointWithColor.color;
        } else {
            // Fall back to the topic colors mapping
            color = topicColors[label.toLowerCase()] || '#999';
        }
        
        datasets.push({
            label: label,
            data: points.map(p => ({ x: p.x, y: p.y })),
            backgroundColor: color,
            pointRadius: 10,
            pointHoverRadius: 12
        });
    });
    
    // Add dataset for user document (initially empty)
    datasets.push({
        label: 'Your Document',
        data: [],
        backgroundColor: '#000000',
        borderColor: '#000000',
        borderWidth: 2,
        pointStyle: 'star',
        pointRadius: 15,
        pointHoverRadius: 18,
        z: 10  // Ensure it's on top of other points
    });
    
    userDocumentPoint = datasets[datasets.length - 1];
    
    // Chart configuration
    topicChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'UMAP Dimension 1'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'UMAP Dimension 2'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const point = context.raw;
                            return context.dataset.label;
                        }
                    }
                }
            }
        }
    });
}

// Update chart with document position
function updateChartWithDocument(position) {
    console.log("Updating document position:", position);
    
    // Update user document point data
    userDocumentPoint.data = [{ x: position.x, y: position.y }];
    
    // Force the point to be visible by setting these properties directly on the dataset
    // userDocumentPoint.backgroundColor = position.color || '#000000';
    userDocumentPoint.backgroundColor = '#000000';
    userDocumentPoint.borderColor = '#000000';
    userDocumentPoint.borderWidth = 2;
    userDocumentPoint.pointStyle = 'triangle';
    userDocumentPoint.radius = 15;
    userDocumentPoint.hoverRadius = 18;
    
    // Update the chart
    topicChart.update();
}

// Toggle example documents
showExamplesBtn.addEventListener('click', () => {
    examplesContainer.style.display = examplesContainer.style.display === 'none' ? 'block' : 'none';
});

// Handle example item clicks
exampleItems.forEach(item => {
    item.addEventListener('click', () => {
        documentInput.value = item.getAttribute('data-full-text');
        examplesContainer.style.display = 'none';
    });
});

// Clear button
clearBtn.addEventListener('click', () => {
    documentInput.value = '';
    errorAlert.style.display = 'none';
    
    // Reset results
    initialState.style.display = 'block';
    loadingState.style.display = 'none';
    resultsState.style.display = 'none';
    
    // Reset chart
    userDocumentPoint.data = [];
    topicChart.update();
});

// Classify document
classifyBtn.addEventListener('click', async () => {
    const text = documentInput.value.trim();
    if (!text) {
        errorAlert.textContent = 'Please enter some text or select an example.';
        errorAlert.style.display = 'block';
        return;
    }
    
    // Show loading state
    errorAlert.style.display = 'none';
    initialState.style.display = 'none';
    loadingState.style.display = 'block';
    resultsState.style.display = 'none';
    
    // Update button
    classifyBtn.disabled = true;
    classifyBtnText.textContent = 'Processing...';
    classifySpinner.style.display = 'inline-block';
    
    try {
        // Call API
        const response = await fetch('/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            throw new Error(`Classification failed: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        console.error('Error:', error);
        errorAlert.textContent = error.message || 'An error occurred during classification.';
        errorAlert.style.display = 'block';
        
        // Show initial state
        initialState.style.display = 'block';
        loadingState.style.display = 'none';
        resultsState.style.display = 'none';
    }
    
    // Reset button
    classifyBtn.disabled = false;
    classifyBtnText.textContent = 'Classify Document';
    classifySpinner.style.display = 'none';
});

// Display classification results
function displayResults(result) {
    // Hide loading, show results
    loadingState.style.display = 'none';
    resultsState.style.display = 'block';
    
    // Update top topic
    const topTopic = result.topTopic;
    topTopicLabel.textContent = topTopic.label;
    
    // Get color with case-insensitive lookup
    const topicColor = topTopic.color || 
                      topicColors[topTopic.label.toLowerCase()] || 
                      '#333';
    
    topTopicLabel.style.color = topicColor;
    topTopicConfidence.textContent = `${Math.round(topTopic.probability * 100)}% confidence`;
    
    // Update confidence scores
    confidenceScores.innerHTML = '';
    result.scores.forEach(score => {
        confidenceScores.appendChild(createScoreBar(score));
    });
    
    // Update key terms
    keyTerms.innerHTML = '';
    const terms = topTopic.terms || [];
    terms.forEach(term => {
        const badge = document.createElement('span');
        badge.className = 'term-badge';
        badge.textContent = term;
        keyTerms.appendChild(badge);
    });
    
    // Update chart
    updateChartWithDocument(result.position);
}

// Initialize the chart when the page loads
document.addEventListener('DOMContentLoaded', () => {
    initChart();
});