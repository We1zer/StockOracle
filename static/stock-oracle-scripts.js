document.addEventListener('DOMContentLoaded', function() {
    // Initialize the SVG background elements
    initBackgroundSVG();
    
    // Mobile Navigation Toggle
    initMobileNav();
    
    // Chart Animation & Interactions
    initChartInteractions();
    
    // Scroll animations
    initScrollAnimations();
    
    // Data loading animations
    animateDataCards();
});

/**
 * Initialize the SVG background with grid lines and chart patterns
 */
function initBackgroundSVG() {
    // Create vertical grid lines
    const verticalLines = document.getElementById('verticalLines');
    const horizontalLines = document.getElementById('horizontalLines');
    const chartPatterns = document.getElementById('chartPatterns');
    
    if (!verticalLines || !horizontalLines || !chartPatterns) return;
    
    // Create vertical lines
    for (let i = 0; i < 50; i++) {
        const x = i * 30;
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x);
        line.setAttribute('y1', 0);
        line.setAttribute('x2', x);
        line.setAttribute('y2', '100%');
        verticalLines.appendChild(line);
    }
    
    // Create horizontal lines
    for (let i = 0; i < 30; i++) {
        const y = i * 30;
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', 0);
        line.setAttribute('y1', y);
        line.setAttribute('x2', '100%');
        line.setAttribute('y2', y);
        horizontalLines.appendChild(line);
    }
    
    // Create animated chart patterns
    createChartPattern(chartPatterns, 100, 100, 400, 200);
    createChartPattern(chartPatterns, 600, 300, 300, 150);
    createChartPattern(chartPatterns, 200, 500, 350, 180);
    createChartPattern(chartPatterns, 800, 150, 250, 120);
}

/**
 * Create an animated chart pattern in SVG
 */
function createChartPattern(container, x, y, width, height) {
    const points = [];
    const numPoints = 10;
    const stepX = width / numPoints;
    
    // Generate random points for the chart
    for (let i = 0; i <= numPoints; i++) {
        const pointX = x + (i * stepX);
        const randomY = y + (Math.random() * height * 0.7) + (height * 0.15);
        points.push(`${pointX},${randomY}`);
    }
    
    // Create the path element
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', `M${points.join(' L')}`);
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', '#3A0CA3');
    path.setAttribute('stroke-width', '2');
    path.setAttribute('stroke-linecap', 'round');
    path.setAttribute('stroke-linejoin', 'round');
    container.appendChild(path);
    
    // Add animation
    const animate = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
    animate.setAttribute('attributeName', 'stroke-dashoffset');
    animate.setAttribute('from', '1000');
    animate.setAttribute('to', '0');
    animate.setAttribute('dur', '3s');
    animate.setAttribute('begin', '0s');
    animate.setAttribute('fill', 'freeze');
    path.appendChild(animate);
    
    // Set stroke-dasharray
    path.setAttribute('stroke-dasharray', '5,5');
}

/**
 * Initialize the mobile navigation
 */
function initMobileNav() {
    const menuToggle = document.getElementById('menuToggle');
    const closeNav = document.getElementById('closeNav');
    const mobileNavOverlay = document.getElementById('mobileNavOverlay');
    
    if (!menuToggle || !closeNav || !mobileNavOverlay) return;
    
    menuToggle.addEventListener('click', function() {
        mobileNavOverlay.classList.add('active');
        document.body.style.overflow = 'hidden';
    });
    
    closeNav.addEventListener('click', function() {
        mobileNavOverlay.classList.remove('active');
        document.body.style.overflow = '';
    });
    
    // Close mobile nav when clicking outside
    mobileNavOverlay.addEventListener('click', function(event) {
        if (event.target === mobileNavOverlay) {
            mobileNavOverlay.classList.remove('active');
            document.body.style.overflow = '';
        }
    });
    
    // Handle mobile nav links
    const mobileNavLinks = document.querySelectorAll('.mobile-nav a');
    mobileNavLinks.forEach(link => {
        link.addEventListener('click', function() {
            mobileNavOverlay.classList.remove('active');
            document.body.style.overflow = '';
        });
    });
}

/**
 * Initialize chart interactions
 */
function initChartInteractions() {
    const chartCards = document.querySelectorAll('.chart-card');
    
    chartCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            const chartImg = this.querySelector('img');
            if (chartImg) {
                chartImg.style.transform = 'scale(1.05)';
                chartImg.style.transition = 'transform 0.3s ease';
            }
        });
        
        card.addEventListener('mouseleave', function() {
            const chartImg = this.querySelector('img');
            if (chartImg) {
                chartImg.style.transform = 'scale(1)';
            }
        });
    });
}

/**
 * Initialize scroll animations
 */
function initScrollAnimations() {
    const animateElements = document.querySelectorAll('.chart-card, .prediction-card, .sentiment-card, .forecast-card, .tweets-card, .recommendation-card');
    
    // Add initial classes
    animateElements.forEach(el => {
        el.classList.add('animate-on-scroll');
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    });
    
    // Check if element is in viewport and animate it
    function checkScroll() {
        animateElements.forEach(el => {
            const rect = el.getBoundingClientRect();
            const windowHeight = window.innerHeight || document.documentElement.clientHeight;
            
            if (rect.top <= windowHeight * 0.85) {
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }
        });
    }
    
    // Initial check
    checkScroll();
    
    // Add scroll event listener
    window.addEventListener('scroll', checkScroll);
}

/**
 * Animate data cards on page load
 */
function animateDataCards() {
    const dataCards = document.querySelectorAll('.stock-data-cards .card');
    
    dataCards.forEach((card, index) => {
        setTimeout(() => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            
            setTimeout(() => {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 100);
        }, index * 100);
    });
}

/**
 * Enhance the data cards with hover effects
 */
document.addEventListener('mouseover', function(e) {
    const card = e.target.closest('.card');
    if (card) {
        const icon = card.querySelector('.card-icon i');
        if (icon) {
            icon.style.transform = 'scale(1.2) rotate(5deg)';
            icon.style.transition = 'transform 0.3s ease';
        }
    }
});

document.addEventListener('mouseout', function(e) {
    const card = e.target.closest('.card');
    if (card) {
        const icon = card.querySelector('.card-icon i');
        if (icon) {
            icon.style.transform = 'scale(1) rotate(0deg)';
        }
    }
});

/**
 * Add animations to prediction cards
 */
function animatePredictionCards() {
    const predictionCards = document.querySelectorAll('.prediction-card');
    
    predictionCards.forEach((card, index) => {
        // Add staggered animation
        setTimeout(() => {
            card.classList.add('animated');
        }, index * 200);
    });
}

/**
 * Animate the recommendation section
 */
function animateRecommendation() {
    const recommendationCard = document.querySelector('.recommendation-card');
    if (!recommendationCard) return;
    
    // Add pulse animation
    function pulseAnimation() {
        recommendationCard.classList.add('pulse');
        
        setTimeout(() => {
            recommendationCard.classList.remove('pulse');
        }, 1000);
    }
    
    // Initial animation after page load
    setTimeout(pulseAnimation, 2000);
    
    // Repeating animation
    setInterval(pulseAnimation, 10000);
}

/**
 * Handle Twitter feed interactions
 */
function initTweetInteractions() {
    const tweetItems = document.querySelectorAll('.tweet-item');
    
    tweetItems.forEach(tweet => {
        tweet.addEventListener('click', function() {
            // Toggle expanded class
            this.classList.toggle('expanded');
            
            const content = this.querySelector('.tweet-content p');
            if (content) {
                if (this.classList.contains('expanded')) {
                    content.style.maxHeight = content.scrollHeight + 'px';
                } else {
                    content.style.maxHeight = '100px';
                }
            }
        });
    });
}

/**
 * Add counter animation to the forecast items
 */
function animateForecastPrices() {
    const forecastPrices = document.querySelectorAll('.forecast-price');
    
    forecastPrices.forEach(price => {
        const finalValue = parseFloat(price.textContent.replace(/[^0-9.-]+/g, ''));
        const duration = 1500; // milliseconds
        const frameRate = 60;
        const frameDuration = 1000 / frameRate;
        const totalFrames = Math.round(duration / frameDuration);
        
        let currentFrame = 0;
        const initialValue = finalValue * 0.8; // Start at 80% of final value
        const valueIncrement = (finalValue - initialValue) / totalFrames;
        
        price.textContent = initialValue.toFixed(2);
        
        const counter = setInterval(() => {
            currentFrame++;
            const currentValue = initialValue + (valueIncrement * currentFrame);
            
            price.textContent = currentValue.toFixed(2);
            
            if (currentFrame >= totalFrames) {
                clearInterval(counter);
                price.textContent = finalValue.toFixed(2);
            }
        }, frameDuration);
    });
}

/**
 * Initialize chart tooltips and info boxes
 */
function initChartTooltips() {
    const chartHeaders = document.querySelectorAll('.chart-header');
    
    chartHeaders.forEach(header => {
        // Create info icon
        const infoIcon = document.createElement('i');
        infoIcon.className = 'fas fa-info-circle';
        infoIcon.style.marginLeft = 'auto';
        infoIcon.style.cursor = 'pointer';
        infoIcon.style.opacity = '0.8';
        infoIcon.style.transition = 'opacity 0.3s ease';
        
        // Add tooltip
        infoIcon.setAttribute('title', 'Click for more information about this chart');
        
        // Append to header
        header.appendChild(infoIcon);
        
        // Add hover effect
        infoIcon.addEventListener('mouseover', function() {
            this.style.opacity = '1';
        });
        
        infoIcon.addEventListener('mouseout', function() {
            this.style.opacity = '0.8';
        });
        
        // Add click event
        infoIcon.addEventListener('click', function() {
            const chartType = header.querySelector('h3').textContent;
            showChartInfo(chartType);
        });
    });
}

/**
 * Show chart information popup
 */
function showChartInfo(chartType) {
    // Create modal overlay
    const overlay = document.createElement('div');
    overlay.className = 'chart-info-overlay';
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    overlay.style.display = 'flex';
    overlay.style.justifyContent = 'center';
    overlay.style.alignItems = 'center';
    overlay.style.zIndex = '3000';
    
    // Create modal content
    const modal = document.createElement('div');
    modal.className = 'chart-info-modal';
    modal.style.backgroundColor = 'white';
    modal.style.borderRadius = '16px';
    modal.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.2)';
    modal.style.maxWidth = '600px';
    modal.style.width = '90%';
    modal.style.maxHeight = '80vh';
    modal.style.overflow = 'auto';
    modal.style.position = 'relative';
    modal.style.animation = 'fadeIn 0.3s ease';
    
    // Create header
    const header = document.createElement('div');
    header.style.padding = '20px';
    header.style.borderBottom = '1px solid #eee';
    header.style.display = 'flex';
    header.style.justifyContent = 'space-between';
    header.style.alignItems = 'center';
    
    const title = document.createElement('h3');
    title.textContent = 'About ' + chartType;
    title.style.margin = '0';
    title.style.color = '#2D46B9';
    
    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '&times;';
    closeBtn.style.backgroundColor = 'transparent';
    closeBtn.style.border = 'none';
    closeBtn.style.fontSize = '24px';
    closeBtn.style.cursor = 'pointer';
    closeBtn.style.color = '#333';
    
    header.appendChild(title);
    header.appendChild(closeBtn);
    
    // Create body
    const body = document.createElement('div');
    body.style.padding = '20px';
    
    // Add appropriate content based on chart type
    const content = document.createElement('div');
    
    if (chartType.includes('TRENDS')) {
        content.innerHTML = `
            <p>This chart displays the recent price movements for the selected stock. The blue line represents the closing price over time.</p>
            <ul style="margin-top: 15px; padding-left: 20px;">
                <li>Upward trends indicate positive price movement</li>
                <li>Downward trends indicate negative price movement</li>
                <li>Flat lines indicate stability in the stock price</li>
            </ul>
            <p style="margin-top: 15px;">The chart helps you visualize the volatility and direction of the stock price over the recent period.</p>
        `;
    } else if (chartType.includes('ARIMA')) {
        content.innerHTML = `
            <p>ARIMA (AutoRegressive Integrated Moving Average) is a statistical analysis model used for forecasting time series data.</p>
            <p style="margin-top: 15px;">This chart shows how accurately the ARIMA model has predicted past prices compared to actual prices:</p>
            <ul style="margin-top: 15px; padding-left: 20px;">
                <li>Orange line: Actual historical prices</li>
                <li>Blue line: ARIMA model predictions</li>
            </ul>
            <p style="margin-top: 15px;">The closer the orange line follows the blue line, the more accurate the model has been historically.</p>
        `;
    } else if (chartType.includes('LSTM')) {
        content.innerHTML = `
            <p>LSTM (Long Short-Term Memory) is a type of recurrent neural network capable of learning long-term dependencies in sequential data.</p>
            <p style="margin-top: 15px;">This chart compares the LSTM model's predictions with actual stock prices:</p>
            <ul style="margin-top: 15px; padding-left: 20px;">
                <li>Orange line: Actual historical prices</li>
                <li>Blue line: LSTM model predictions</li>
            </ul>
            <p style="margin-top: 15px;">LSTM models are particularly good at capturing complex patterns in stock price movements.</p>
        `;
    } else if (chartType.includes('LINEAR REGRESSION')) {
        content.innerHTML = `
            <p>Linear Regression is a statistical approach for modeling the relationship between a dependent variable and one or more independent variables.</p>
            <p style="margin-top: 15px;">This chart shows how the Linear Regression model predictions compare to actual prices:</p>
            <ul style="margin-top: 15px; padding-left: 20px;">
                <li>Orange line: Actual historical prices</li>
                <li>Blue line: Linear Regression model predictions</li>
            </ul>
            <p style="margin-top: 15px;">Linear Regression works best when there is a linear relationship in the data.</p>
        `;
    } else {
        content.innerHTML = `
            <p>This chart provides valuable insights into the performance and predictions for the selected stock.</p>
            <p style="margin-top: 15px;">Our models use a combination of technical indicators, historical data, and advanced algorithms to generate predictions.</p>
        `;
    }
    
    body.appendChild(content);
    
    // Assemble modal
    modal.appendChild(header);
    modal.appendChild(body);
    overlay.appendChild(modal);
    
    // Add to body
    document.body.appendChild(overlay);
    
    // Add close event
    closeBtn.addEventListener('click', function() {
        document.body.removeChild(overlay);
    });
    
    // Close when clicking overlay
    overlay.addEventListener('click', function(e) {
        if (e.target === overlay) {
            document.body.removeChild(overlay);
        }
    });
}

// Run animations when page has loaded
window.addEventListener('load', function() {
    animatePredictionCards();
    animateRecommendation();
    initTweetInteractions();
    animateForecastPrices();
    initChartTooltips();
    
    // Add keyframe animation for pulse effect
    const style = document.createElement('style');
    style.innerHTML = `
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        
        .pulse {
            animation: pulse 1s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    `;
    document.head.appendChild(style);
});

// Add a light data update simulation
function simulateDataUpdates() {
    const stockPrices = document.querySelectorAll('.card-info h2');
    
    setInterval(() => {
        // Randomly select a price to update
        const randomIndex = Math.floor(Math.random() * stockPrices.length);
        const selectedPrice = stockPrices[randomIndex];
        
        if (selectedPrice) {
            // Get current value
            let currentValue = parseFloat(selectedPrice.textContent.replace(/[^0-9.-]+/g, ''));
            
            // Generate small random change
            const change = (Math.random() - 0.5) * 2; // Between -1 and 1
            const newValue = (currentValue + change).toFixed(2);
            
            // Flash animation
            selectedPrice.style.transition = 'color 0.5s ease';
            selectedPrice.style.color = change >= 0 ? '#4BB543' : '#DC3545';
            
            // Update value
            selectedPrice.textContent = newValue;
            
            // Reset color
            setTimeout(() => {
                selectedPrice.style.color = '';
            }, 800);
        }
    }, 8000); // Every 8 seconds
}

// Initialize simulation after a delay
setTimeout(simulateDataUpdates, 10000);