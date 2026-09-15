const API_BASE_URL = `${window.location.protocol}//${window.location.hostname}:8000`;


/* =========================
   REGISTER
========================= */

const registerForm =
    document.getElementById("registerForm");

if (registerForm) {

    registerForm.addEventListener(
        "submit",
        async function(event) {

            event.preventDefault();

            const name =
                document.getElementById("name").value;

            const email =
                document.getElementById("registerEmail").value;

            const password =
                document.getElementById("registerPassword").value;

            const confirmPassword =
                document.getElementById(
                    "confirmPassword"
                ).value;

            if (password !== confirmPassword) {
                alert("Passwords do not match.");
                return;
            }

            try {

                const response =
                    await fetch(
                        API_BASE_URL + "/register?" +
                        "name=" +
                        encodeURIComponent(name) +
                        "&email=" +
                        encodeURIComponent(email) +
                        "&password=" +
                        encodeURIComponent(password),
                        {
                            method: "POST"
                        }
                    );

                const result =
                    await response.json();

                if (!result.success) {
                    alert(result.message);
                    return;
                }

                window.location.href =
                    "login.html";

            } catch (error) {

                console.error(error);

                alert(
                    "Unable to connect to the backend."
                );
            }
        }
    );
}


/* =========================
   LOGIN
========================= */

const loginForm =
    document.getElementById("loginForm");

if (loginForm) {

    loginForm.addEventListener(
        "submit",
        async function(event) {

            event.preventDefault();

            const email =
                document.getElementById(
                    "loginEmail"
                ).value;

            const password =
                document.getElementById(
                    "loginPassword"
                ).value;

            try {

                const response =
                    await fetch(
                        API_BASE_URL + "/login?" +
                        "email=" +
                        encodeURIComponent(email) +
                        "&password=" +
                        encodeURIComponent(password),
                        {
                            method: "POST"
                        }
                    );

                const result =
                    await response.json();

                if (!result.success) {
                    alert(result.message);
                    return;
                }

                window.location.href =
                    "dashboard.html";

            } catch (error) {

                console.error(error);

                alert(
                    "Unable to connect to the backend."
                );
            }
        }
    );
}


/* =========================
   MRI UPLOAD
========================= */

const uploadBox =
    document.getElementById("uploadBox");

const imageInput =
    document.getElementById("imageInput");

const imagePreview =
    document.getElementById("imagePreview");

const originalImage =
    document.getElementById("originalImage");


if (uploadBox && imageInput) {

    uploadBox.addEventListener(
        "click",
        function() {

            imageInput.click();

        }
    );


    imageInput.addEventListener(
        "change",
        function() {

            const file =
                imageInput.files[0];


            if (!file) {
                return;
            }


            if (!file.type.startsWith("image/")) {

                alert(
                    "Please select an image file."
                );

                return;
            }


            const reader =
                new FileReader();


            reader.onload =
                function(event) {

                    imagePreview.src =
                        event.target.result;

                    imagePreview.style.display =
                        "block";

                    originalImage.src =
                        event.target.result;

                    originalImage.style.display =
                        "block";

                };


            reader.readAsDataURL(file);

        }
    );
}


/* =========================
   ANALYZE
========================= */

const analyzeBtn =
    document.getElementById("analyzeBtn");


if (analyzeBtn) {

    analyzeBtn.addEventListener(
        "click",
        async function() {

            if (
                !imageInput ||
                !imageInput.files.length
            ) {

                alert(
                    "Please upload an MRI image first."
                );

                return;
            }


            const selectedFile =
                imageInput.files[0];


            const formData =
                new FormData();

            formData.append(
                "file",
                selectedFile
            );


            try {

                const response =
                    await fetch(
                        API_BASE_URL + "/predict",
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                if (!response.ok) {

                    throw new Error(
                        "Prediction request failed."
                    );

                }


                const result =
                    await response.json();


                const limeResponse =
                    await fetch(
                        API_BASE_URL + "/explain",
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                if (!limeResponse.ok) {

                    throw new Error(
                        "LIME explanation failed."
                    );

                }


                const limeBlob =
                    await limeResponse.blob();


                const limeUrl =
                    URL.createObjectURL(
                        limeBlob
                    );


                const limeImage =
                    document.getElementById(
                        "limeImage"
                    );


                limeImage.src =
                    limeUrl;

                limeImage.style.display =
                    "block";


                document.getElementById(
                    "predictedClass"
                ).textContent =
                    result.predicted_class;


                document.getElementById(
                    "confidence"
                ).textContent =
                    (result.confidence * 100)
                    .toFixed(2) + "%";


                document.getElementById(
                    "riskClassification"
                ).textContent =
                    "MODEL-BASED";


                document.getElementById(
                    "summaryClass"
                ).textContent =
                    result.predicted_class;


                document.getElementById(
                    "summaryConfidence"
                ).textContent =
                    (result.confidence * 100)
                    .toFixed(2) + "%";


                document.getElementById(
                    "summaryClassification"
                ).textContent =
                    "MODEL-BASED";


                const probabilities =
                    result.probabilities;


                console.log(
                    "PROBABILITIES:",
                    probabilities
                );


                console.log(
                    "KEYS:",
                    Object.keys(probabilities)
                );


                const ctx =
                    document
                        .getElementById(
                            "probabilityChart"
                        )
                        .getContext("2d");


                if (window.myprobabilityChart) {

                    window.myprobabilityChart.destroy();

                }


                window.myprobabilityChart =
                    new Chart(ctx, {

                        type: "bar",

                        data: {

                            labels: [
                                "Glioma",
                                "Meningioma",
                                "No Tumor",
                                "Pituitary"
                            ],

                            datasets: [{

                                label:
                                    "Probability",

                                data: [

                                    probabilities.glioma *
                                        100,

                                    probabilities.meningioma *
                                        100,

                                    probabilities["No-tumor"] *
                                        100,

                                    probabilities.pituitary *
                                        100

                                ]

                            }]

                        },

                        options: {

                            responsive: true,

                            scales: {

                                y: {

                                    beginAtZero: true,

                                    max: 100

                                }

                            }

                        }

                    });

            }


            catch (error) {

                console.error(error);

                alert(
                    "Unable to connect to the AI backend."
                );

            }

        }
    );
}


/* =========================
   LOGOUT
========================= */

const logoutBtn =
    document.getElementById("logoutBtn");


if (logoutBtn) {

    logoutBtn.addEventListener(
        "click",
        function(event) {

            event.preventDefault();

            window.location.href =
                "index.html";

        }
    );
}