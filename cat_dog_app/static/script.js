document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const previewImage = document.getElementById('previewImage');
    const uploadContent = document.querySelector('.upload-content');
    const classifyBtn = document.getElementById('classifyBtn');
    const btnText = classifyBtn.querySelector('span');
    const btnLoader = document.getElementById('btnLoader');
    const resultCard = document.getElementById('resultCard');
    const resultTitle = document.getElementById('resultTitle');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    const errorMsg = document.getElementById('errorMsg');

    let currentFile = null;

    // Handle Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
    });

    uploadArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    uploadArea.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', function () {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;

        const file = files[0];
        if (!file.type.startsWith('image/')) {
            showError('Please upload an image file.');
            return;
        }

        currentFile = file;
        hideError();
        resultCard.classList.add('hidden');

        // Setup preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.classList.remove('hidden');
            uploadContent.style.opacity = '0';
            classifyBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    classifyBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI Loading State
        classifyBtn.disabled = true;
        btnText.textContent = 'Analyzing...';
        btnLoader.classList.remove('hidden');
        resultCard.classList.add('hidden');
        hideError();

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok && !data.error) {
                showResult(data.prediction, data.confidence);
            } else {
                throw new Error(data.error || 'Failed to classify image.');
            }
        } catch (error) {
            showError(error.message);
        } finally {
            classifyBtn.disabled = false;
            btnText.textContent = 'Classify Image';
            btnLoader.classList.add('hidden');
        }
    });

    function showResult(prediction, confidence) {
        resultTitle.textContent = `It's a ${prediction}!`;
        confidenceText.textContent = `${confidence.toFixed(1)}%`;
        resultCard.classList.remove('hidden');

        // Color based on prediction just for flair
        if (prediction.toLowerCase() === 'cat') {
            resultTitle.style.color = '#60A5FA'; // Blue for cat
            confidenceFill.style.background = 'linear-gradient(90deg, #3B82F6, #60A5FA)';
        } else {
            resultTitle.style.color = '#F472B6'; // Pink for dog
            confidenceFill.style.background = 'linear-gradient(90deg, #EC4899, #F472B6)';
        }

        // Trigger animation
        setTimeout(() => {
            confidenceFill.style.width = `${confidence}%`;
        }, 100);
    }

    function showError(msg) {
        errorMsg.textContent = msg;
        errorMsg.classList.remove('hidden');
    }

    function hideError() {
        errorMsg.classList.add('hidden');
        errorMsg.textContent = '';
    }
});
