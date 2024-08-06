document.addEventListener('DOMContentLoaded', function() {
  const form = document.getElementById('scorer-form');
  const resultDiv = document.getElementById('result');
  const scoreSpan = document.getElementById('score');
  const loadingDiv = document.getElementById('loading');
  const errorDiv = document.getElementById('error');
  const summaryTextarea = document.getElementById('summary');
  const summaryImageInput = document.getElementById('summary_image');
  const extractTextBtn = document.getElementById('extract-text-btn');
  const extractedTextDiv = document.getElementById('extracted-text');
  const extractedTextContent = document.getElementById('extracted-text-content');

  extractTextBtn.addEventListener('click', async function() {
      if (summaryImageInput.files.length === 0) {
          showError('Please select an image file first.');
          return;
      }

      showLoading();
      hideError();

      const imageFormData = new FormData();
      imageFormData.append('summary_image', summaryImageInput.files[0]);

      try {
          const response = await fetch('/extract_text', {
              method: 'POST',
              body: imageFormData
          });

          const data = await response.json();

          if (!response.ok) {
              throw new Error(data.error || 'Failed to extract text from image');
          }

          summaryTextarea.value = data.text;
          extractedTextContent.textContent = data.text;
          extractedTextDiv.classList.remove('hidden');
      } catch (error) {
          showError(error.message);
      } finally {
          hideLoading();
      }
  });

  form.addEventListener('submit', async function(e) {
      e.preventDefault();
      resultDiv.classList.add('hidden');
      showLoading();
      hideError();

      const formData = new FormData(form);

      try {
          const scoreResponse = await fetch('/score', {
              method: 'POST',
              body: formData
          });

          if (!scoreResponse.ok) {
              throw new Error('Failed to calculate score');
          }

          const scoreData = await scoreResponse.json();
          scoreSpan.textContent = scoreData.score;
          resultDiv.classList.remove('hidden');
      } catch (error) {
          showError(error.message);
      } finally {
          hideLoading();
      }
  });

  function showLoading() {
      loadingDiv.classList.remove('hidden');
  }

  function hideLoading() {
      loadingDiv.classList.add('hidden');
  }

  function showError(message) {
      errorDiv.textContent = message;
      errorDiv.classList.remove('hidden');
  }

  function hideError() {
      errorDiv.classList.add('hidden');
  }
});