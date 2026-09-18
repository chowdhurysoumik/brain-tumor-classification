# 🧠 Brain Tumor AI — Full-Stack MRI Classification & Explainable AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/AWS-EC2-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white" alt="AWS EC2"/>
</p>

<p align="center">
  <strong>A complete end-to-end AI application for brain MRI classification, explainable predictions, authentication, containerization, and cloud deployment.</strong>
</p>

---

## 📌 Overview

A full-stack AI platform for **multi-class brain tumor classification from MRI scans**.

The system combines:

* 🧠 **TensorFlow/Keras CNN** for MRI classification
* 🔍 **LIME Explainable AI (XAI)** for model interpretability
* ⚡ **FastAPI REST backend**
* 🎨 **Nginx-served frontend**
* 🔐 **SQLAlchemy-based authentication**
* 🗄️ **Persistent SQLite database**
* 🐳 **Docker & Docker Compose**
* 📦 **Docker Hub image distribution**
* ☁️ **AWS EC2 cloud deployment**



---

## 💡 Quick Overview

| 🏷️ Domain               | 🚀 Key Highlights                                                                                   |
| :----------------------- | :-------------------------------------------------------------------------------------------------- |
| 🧠 **Machine Learning**  | Custom CNN, data augmentation, multi-class Softmax classification                                   |
| 🔍 **Explainable AI**    | LIME-based region-of-interest heatmaps                                                              |
| ⚡ **Backend & Database** | FastAPI REST API, OpenAPI/Swagger, SQLAlchemy ORM, SQLite                                           |
| 🎨 **Frontend UI**       | Responsive Vanilla JS dashboard, real-time image preview, probability visualization, authentication |
| 🐳 **DevOps & Cloud**    | Docker multi-container architecture, Docker Hub, AWS EC2                                            |

### 🎯 Supported Classes

* 🧠 Glioma
* 🧠 Meningioma
* 🧠 Pituitary
* 🟢 No Tumor

---

# 🏗️ System Architecture

The application follows a **Frontend → API → AI Inference → Database / Explainability** architecture.

```mermaid
flowchart TD
    A(["🌐 Client"]) --> B(["👤 User Browser<br/>HTML5 · CSS3 · JavaScript"])

    B -->|HTTP :8080| C(["🎨 Frontend Container<br/>Nginx Alpine"])

    C -->|REST API :8000| D(["⚡ Backend Container<br/>FastAPI"])

    D --> E(["🧠 TensorFlow CNN"])
    D --> F(["🔍 LIME Engine"])
    D --> G(["🗄️ SQLAlchemy + SQLite"])

    E --> H(["🎯 AI Prediction"])
    F --> I(["💡 Explanation"])

    H --> J(["📊 Results"])
    I --> J

    K(["☁️ AWS EC2 Instance"]) --> C
    K --> D

    classDef default fill:#ffffff,stroke:#64748b,stroke-width:2px,color:#0f172a;
    classDef cloud fill:#f8fafc,stroke:#475569,stroke-width:2px,color:#0f172a;

    class A,B,C,D,E,F,G,H,I,J,K default;
```

---

# 🔄 Application Workflow

The application follows an end-to-end workflow from authentication to MRI classification and explainable prediction.

```mermaid
flowchart TD
    A(["👤 User<br/>Opens Web App"])
    B(["🔐 Register / Login<br/>Authentication"])
    C(["🧠 Upload MRI Image<br/>Preview Scan"])
    D(["🎨 Frontend JavaScript<br/>Sends API Request"])
    E(["⚡ FastAPI Backend<br/>/predict"])
    F(["🛠️ Image Processing<br/>Resize + Normalize"])
    G(["🧠 TensorFlow CNN<br/>Model Inference"])
    H(["📊 Classification<br/>Glioma · Meningioma<br/>Pituitary · No Tumor"])

    I(["📊 Class Probabilities<br/>Visualization"])
    J(["🔍 LIME Analysis<br/>Explain Prediction"])
    K(["🎯 Results Dashboard<br/>Prediction · Probabilities<br/>LIME Explanation"])

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H

    H --> I
    H --> J

    I --> K
    J --> K

    classDef default fill:#ffffff,stroke:#64748b,stroke-width:2px,color:#0f172a;
    class A,B,C,D,E,F,G,H,I,J,K default;
```

---

# 🧠 Machine Learning Pipeline

The application uses a **TensorFlow/Keras CNN** to classify brain MRI scans into four categories:

**Glioma · Meningioma · Pituitary · No Tumor**

```mermaid
flowchart TD
    A(["🧠 MRI Image"])
    B(["📐 Resize<br/>224 × 224"])
    C(["🔢 Normalize<br/>0–1"])
    D(["🔄 Data Augmentation"])
    E(["🧠 CNN Model"])
    F(["📊 Softmax Probabilities"])
    G(["🎯 Predicted Class"])

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G

    classDef default fill:#ffffff,stroke:#64748b,stroke-width:2px,color:#0f172a;
    class A,B,C,D,E,F,G default;
```

---

# 🔍 Explainable AI — LIME

To improve model interpretability, the application integrates **LIME (Local Interpretable Model-Agnostic Explanations)**.

After generating a prediction, the system analyzes the MRI image and identifies regions that contributed most to the model's decision.

```mermaid
flowchart TD
    A(["🧠 MRI Image"])
    B(["🧠 CNN Prediction"])
    C(["🔍 LIME Analysis"])
    D(["🟥 Important Image Regions"])
    E(["💡 Visual Explanation"])

    A --> B
    B --> C
    C --> D
    D --> E

    classDef default fill:#ffffff,stroke:#64748b,stroke-width:2px,color:#0f172a;
    class A,B,C,D,E default;
```

### ✨ Explainability Flow

```mermaid
flowchart TD
    A(["🎯 Prediction"])
    B(["🔍 LIME Perturbation"])
    C(["🧠 Model Responses"])
    D(["📍 Important Regions"])
    E(["🔥 Heatmap / Explanation"])

    A --> B
    B --> C
    C --> D
    D --> E

    classDef default fill:#ffffff,stroke:#64748b,stroke-width:2px,color:#0f172a;
    class A,B,C,D,E default;
```

---

# ⚡ FastAPI Backend

The application uses **FastAPI** to provide a lightweight REST API for authentication, MRI prediction, and explainable AI.

## 🔌 API Endpoints

| Method | Endpoint    | Description                              |
| :----: | :---------- | :--------------------------------------- |
|  `GET` | `/`         | 🩺 API health/status check               |
| `POST` | `/register` | 👤 Register a new user                   |
| `POST` | `/login`    | 🔐 Authenticate an existing user         |
| `POST` | `/predict`  | 🧠 Classify a brain MRI image            |
| `POST` | `/explain`  | 🔍 Generate a LIME explanation           |
|  `GET` | `/docs`     | 📚 Interactive Swagger API documentation |

## 🛠️ Backend Responsibilities

* 👤 User registration and authentication
* 📤 MRI image upload handling
* 🧠 Brain tumor classification using the trained CNN
* 📊 Prediction probability generation
* 🔍 LIME-based visual explanations
* 🗄️ SQLite database integration using SQLAlchemy
* 🔌 REST API communication with the frontend

FastAPI's built-in **Swagger UI** makes it easy to test and explore the available endpoints during development and deployment.

---

# 🔐 Database & Authentication

The application includes a user authentication system backed by **SQLite** and **SQLAlchemy**.

## 🔄 Authentication Flow

### 👤 Registration

```mermaid
flowchart TD
    A(["👤 User Registration"])
    B(["⚡ FastAPI API"])
    C(["🗄️ SQLite Database"])
    D(["✅ User Account Created"])

    A --> B
    B --> C
    C --> D

    classDef default fill:#ffffff,stroke:#64748b,stroke-width:2px,color:#0f172a;
    class A,B,C,D default;
```

### 🔑 Login

```mermaid
flowchart TD
    A(["👤 User Login"])
    B(["⚡ FastAPI Authentication"])
    C(["🔐 Validate Credentials"])
    D(["🎯 Access Dashboard"])

    A --> B
    B --> C
    C --> D

    classDef default fill:#ffffff,stroke:#64748b,stroke-width:2px,color:#0f172a;
    class A,B,C,D default;
```

---

# 🎨 Frontend

The frontend provides the user-facing interface for authentication, MRI image upload, prediction results, and explainable AI visualization.

## ✨ Features

* 👤 User registration and login
* 🧠 Brain MRI image upload
* ⚡ Real-time communication with the FastAPI backend
* 📊 Predicted tumor class and probability display
* 🔍 LIME explanation visualization
* 🎯 Dashboard-based user experience
* 📱 Responsive interface using HTML and CSS

## 💻 Technology

| Technology     | Purpose                      |
| :------------- | :--------------------------- |
| **HTML5**      | 🧱 Page structure            |
| **CSS3**       | 🎨 Styling and responsive UI |
| **JavaScript** | ⚡ Client-side functionality  |
| **Nginx**      | 🌐 Frontend web server       |

The frontend dynamically determines the backend API host from the current browser hostname, allowing the same frontend build to work across local Docker and AWS EC2 deployments.

---

# 🐳 Docker Architecture

The application is containerized using **Docker**, with separate containers for the backend API and frontend web server.

## 📦 Container Architecture

```mermaid
flowchart TD
    A(["👤 User"])
    B(["🌐 Web Browser"])

    C(["🎨 Frontend Container<br/>Nginx + HTML/CSS/JS<br/>Port: 8080"])
    D(["⚡ Backend Container<br/>FastAPI + TensorFlow<br/>Port: 8000"])

    E(["🗄️ SQLite Database<br/>brain_tumor.db"])
    F(["🧠 CNN Model<br/>best_model.keras"])

    A --> B
    B --> C
    C -->|HTTP API Requests| D

    D --> E
    D --> F

    classDef default fill:#ffffff,stroke:#64748b,stroke-width:2px,color:#0f172a;
    class A,B,C,D,E,F default;
```

---

# 📦 Docker Hub

The containerized application is published to **Docker Hub** so the deployment environment can pull pre-built images without rebuilding the application on the server.

## 🔄 Image Workflow

```mermaid
flowchart TD
    A(["💻 Source Code"])
    B(["🐳 Build Docker Images"])
    C(["📦 Push Images to Docker Hub"])
    D(["☁️ AWS EC2"])
    E(["⬇️ Pull Images"])
    F(["🚀 Run Containers"])

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F

    classDef default fill:#ffffff,stroke:#64748b,stroke-width:2px,color:#0f172a;
    class A,B,C,D,E,F default;
```

---

# ☁️ AWS EC2 Deployment

The application is deployed on **Amazon EC2** using Docker containers.

## 🖥️ Deployment Environment

| Configuration           | Value                 |
| :---------------------- | :-------------------- |
| ☁️ **Cloud Provider**   | AWS                   |
| 🖥️ **Service**         | Amazon EC2            |
| 🐧 **Operating System** | Ubuntu Server         |
| ⚙️ **Instance Type**    | `t3.small`            |
| 🏗️ **Architecture**    | x86_64                |
| 💾 **Storage**          | 20 GiB gp3            |
| 🌏 **Region**           | Asia Pacific (Mumbai) |

## 🏗️ Deployment Architecture

```mermaid
flowchart TD
    A(["☁️ AWS EC2 Instance"])

    B(["⚡ Backend Container<br/>FastAPI + TensorFlow<br/>Port 8000"])
    C(["🎨 Frontend Container<br/>Nginx<br/>Port 8080"])

    D(["🗄️ SQLite Database"])
    E(["🧠 CNN Model"])

    A --> B
    A --> C

    B --> D
    B --> E

    classDef default fill:#ffffff,stroke:#64748b,stroke-width:2px,color:#0f172a;
    class A,B,C,D,E default;
```

---

# 📈 Project Evolution

The project evolved from a standalone CNN-based classification model into a complete full-stack AI application.

## 🔄 Evolution Timeline

```mermaid
flowchart TD
    A(["🧠 CNN Model Development"])
    B(["🩻 Brain MRI Classification"])
    C(["📊 Model Evaluation"])
    D(["⚡ FastAPI Inference Backend"])
    E(["🔍 LIME Explainability"])
    F(["🔐 User Authentication + Database"])
    G(["🎨 Web-Based Frontend"])
    H(["🐳 Docker Containerization"])
    I(["📦 Docker Hub Image Distribution"])
    J(["☁️ AWS EC2 Deployment"])

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J

    classDef default fill:#ffffff,stroke:#64748b,stroke-width:2px,color:#0f172a;
    class A,B,C,D,E,F,G,H,I,J default;
```

---

# 🛠️ Engineering Highlights

This project demonstrates practical **software engineering, machine learning, and cloud deployment skills** alongside the core AI model.

## 🔧 Key Engineering Practices

| Area                       | Implementation                                          |
| :------------------------- | :------------------------------------------------------ |
| ⚡ **Backend**              | RESTful API using FastAPI                               |
| 🧠 **AI Inference**        | TensorFlow/Keras CNN                                    |
| 🔍 **Explainability**      | LIME-based visual explanations                          |
| 🗄️ **Database**           | SQLite + SQLAlchemy                                     |
| 🎨 **Frontend**            | Separate frontend/backend architecture                  |
| 🐳 **Containerization**    | Docker                                                  |
| 🔄 **Orchestration**       | Docker Compose                                          |
| 📦 **Distribution**        | Docker Hub                                              |
| ☁️ **Deployment**          | AWS EC2                                                 |
| 🔐 **Networking**          | AWS Security Groups                                     |
| 💾 **Persistence**         | SQLite database persisted outside backend container     |
| 🛡️ **Repository Hygiene** | `.gitignore` and `.dockerignore`                        |
| 🔬 **ML Integration**      | Original ML work extended into a deployable application |

---

# 🎯 Project Purpose

This project was developed to explore how a **machine learning model can be transformed into a complete, deployable AI application**.

Rather than stopping at model training and evaluation, the project combines:

| Component                   | Technology                     |
| :-------------------------- | :----------------------------- |
| 🧠 **Machine Learning**     | CNN-based MRI classification   |
| 🔍 **Explainable AI**       | LIME-based visual explanations |
| ⚡ **Backend Development**   | FastAPI REST API               |
| 🎨 **Frontend Development** | HTML, CSS, JavaScript          |
| 🔐 **Authentication**       | User registration and login    |
| 🗄️ **Database**            | SQLite with SQLAlchemy         |
| 🐳 **Containerization**     | Docker & Docker Compose        |
| 📦 **Image Distribution**   | Docker Hub                     |
| ☁️ **Cloud Deployment**     | AWS EC2                        |

---

## 🚀 End-to-End Journey

```mermaid
flowchart TD
    A(["🧪 Machine Learning"])
    B(["🧠 CNN Development"])
    C(["🩻 MRI Classification"])
    D(["🔍 XAI / LIME"])
    E(["⚡ FastAPI Backend"])
    F(["🎨 Web Frontend"])
    G(["🔐 Authentication"])
    H(["🗄️ Database"])
    I(["🐳 Dockerization"])
    J(["📦 Docker Hub"])
    K(["☁️ AWS EC2"])
    L(["🚀 Deployed AI Application"])

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L

    classDef default fill:#ffffff,stroke:#64748b,stroke-width:2px,color:#0f172a;
    class A,B,C,D,E,F,G,H,I,J,K,L default;
```

---

## ⭐ Project Stack

```mermaid
flowchart LR
    A(["🧠 AI / ML<br/>TensorFlow · Keras · LIME"])
    B(["⚡ Backend<br/>FastAPI · SQLAlchemy"])
    C(["🎨 Frontend<br/>HTML · CSS · JavaScript"])
    D(["🗄️ Database<br/>SQLite"])
    E(["🐳 DevOps<br/>Docker · Compose · Docker Hub"])
    F(["☁️ Cloud<br/>AWS EC2"])

    A --> B
    B --> C
    B --> D
    B --> E
    E --> F

    classDef default fill:#ffffff,stroke:#64748b,stroke-width:2px,color:#0f172a;
    class A,B,C,D,E,F default;
```

---

## 📌 Final Architecture

```mermaid
flowchart TB
    U(["👤 User"])

    FE(["🎨 Frontend<br/>HTML · CSS · JavaScript<br/>Nginx :8080"])

    API(["⚡ FastAPI Backend<br/>REST API :8000"])

    ML(["🧠 TensorFlow CNN<br/>MRI Classification"])

    XAI(["🔍 LIME<br/>Explainable AI"])

    DB(["🗄️ SQLite<br/>User Data"])

    HUB(["📦 Docker Hub"])

    AWS(["☁️ AWS EC2"])

    RESULT(["🎯 Results Dashboard<br/>Prediction · Probabilities · Explanation"])

    U --> FE
    FE --> API

    API --> ML
    API --> XAI
    API --> DB

    ML --> RESULT
    XAI --> RESULT

    FE --> AWS
    API --> AWS

    HUB --> AWS

    classDef default fill:#ffffff,stroke:#64748b,stroke-width:2px,color:#0f172a;
    class U,FE,API,ML,XAI,DB,HUB,AWS,RESULT default;
```

---

<p align="center">
  <strong>🧠 From MRI Classification → Explainable AI → Full-Stack Application → Docker → AWS 🚀</strong>
</p>


---

# 👨‍💻 Team

## Developers

* **Sourja Goswamy**
* **Soumik Chowdhury**
