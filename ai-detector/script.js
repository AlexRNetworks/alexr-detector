document.addEventListener('DOMContentLoaded', function() {
    const detectButton = document.getElementById('detectButton');
    const inputText = document.getElementById('inputText');
    const resultsDiv = document.getElementById('results');
    const predictionSpan = document.getElementById('prediction');
    const confidenceSpan = document.getElementById('confidence');
    const explanationP = document.getElementById('explanation');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const confidenceBar = document.getElementById('confidenceBar');
    const scoreLabel = document.getElementById('scoreLabel'); // Get the score label


    detectButton.addEventListener('click', async () => {
        const text = inputText.value.trim();

        if (!text) {
            alert('Please enter some text to analyze.');
            return;
        }

        // Show loading spinner, hide results, reset progress bar, and reset label
        loadingSpinner.classList.remove('hidden');
        resultsDiv.classList.add('hidden');
        resultsDiv.classList.remove('visible');
        confidenceBar.value = 0;
        scoreLabel.textContent = "Here's Your Score:"; // Reset label


        try {
            const response = await fetch('/.netlify/functions/api', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // Update UI with results
            const confidencePercentage = (data.confidence * 100).toFixed(1);
            const prediction = confidencePercentage >= 60 ? "Human" : "AI"; // Determine prediction

            predictionSpan.textContent = prediction;
            confidenceSpan.textContent = `${confidencePercentage}%`;
            confidenceBar.value = confidencePercentage;
            explanationP.textContent = data.explanation || "No explanation provided.";

            // Update score label
            scoreLabel.textContent = `Here's Your Score: ${prediction} (${confidencePercentage}%)`;


            resultsDiv.classList.add('visible');
            resultsDiv.classList.remove('hidden');

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during detection. Please try again.');
        } finally {
            loadingSpinner.classList.add('hidden');
        }
    });
});
