const classNames = [
    "Aeroplane", "Automobile", "Bird", "Cat", "Deer", 
    "Dog", "Frog", "Horse", "Ship", "Truck"
];

document.getElementById('drop-area').addEventListener('click', function() {
    document.getElementById('file-input').click();
});  

document.getElementById("file-input").addEventListener("change", function(event) {
    const preview = document.getElementById("image-preview");
    const file = event.target.files[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = "block";
        };
        reader.readAsDataURL(file);
    } else {
        preview.style.display = "none";
    }
});

document.getElementById("upload-form").addEventListener("submit", function(event) {
    event.preventDefault();
    
    let formData = new FormData();
    let fileInput = document.getElementById("file-input").files[0];
    let resultBox = document.getElementById("result-box");

    if (!fileInput) {
        resultBox.innerText = "⚠️ Please select an image.";
        resultBox.style.display = "block";
        resultBox.style.color = "red";
        return;
    }

    formData.append("file", fileInput);
    resultBox.innerText = "⏳ Processing...";
    resultBox.style.display = "block";
    resultBox.style.color = "black";

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const classIndex = parseInt(data.prediction);

        if ((Number.isInteger(classIndex) && classIndex >= 0 && classIndex < classNames.length) && (data.prediction !== undefined)) {
            const className = classNames[classIndex] || "Unknown";
            resultBox.innerText = `✅ Prediction: ${className}`;
            resultBox.style.color = "green";
        } else {
            resultBox.innerText = "❌ Error: Failed to classify image.";
            resultBox.style.color = "red";
        }
    })
    .catch(error => {
        console.error("Error:", error);
        resultBox.innerText = "❌ Error: Server issue, try again.";
        resultBox.style.color = "red";
    });
});
