document.addEventListener('DOMContentLoaded', function() {
    // File upload preview
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    
    if (fileInput && previewContainer && previewImage) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    previewContainer.classList.remove('d-none');
                }
                reader.readAsDataURL(file);
            }
        });
    }

    // Flash message auto-close
    const flashMessages = document.querySelectorAll('.alert');
    flashMessages.forEach(msg => {
        setTimeout(() => {
            msg.classList.add('fade');
            setTimeout(() => msg.remove(), 300);
        }, 5000);
    });
});