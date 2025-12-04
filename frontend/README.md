# Lumos PWA

![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![PWA](https://img.shields.io/badge/PWA-Progressive%20Web%20App-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-UI%20Demo-orange?style=for-the-badge)

A React-based Progressive Web App (PWA) designed for **Lumos**: A Mobile-first AI Image Editor for 2030.

## 🔗 Live Demo

**Access the deployment here:**
 **[https://lumos-team76.vercel.app/](https://lumos-team76.vercel.app/)**

---


## ⚠️ Important Disclaimer: UI Demo Only

**Please note that this deployment is strictly a Frontend User Interface demonstration.**

* **No Connected Backend:** This application is currently running as a standalone client-side build.
* **functionality:** Features requiring server-side processing (such as AI image generation, voice processing, or database storage) are **not active** in this demo.
* **Purpose:** The goal of this deployment is to showcase the visual design, interactions, animations, and PWA capabilities.

---

## 📱 Device Optimization & PWA Installation

This application has been meticulously **optimized for the iPhone 16 Pro Max** form factor. For the intended user experience (full-screen mode, correct aspect ratio, touch optimization, and performance), **it must be installed on your mobile device.**

### How to Install (Required for correct UI rendering)

**iOS (Safari) - Recommended:**
1.  Open [the application URL](https://lumos-team76.vercel.app/) in Safari.
2.  Tap the **Share** button (square with an arrow).
3.  Scroll down and select **"Add to Home Screen"**.
4.  Launch the app from your home screen to view it in full-screen standalone mode.

**Android (Chrome):**
1.  Open the application URL in Chrome.
2.  Tap the menu icon (three dots in the upper right).
3.  Select **"Install App"** or **"Add to Home Screen"**.

---


## 🔧 Architecture & Dependencies

### Voice Recognition API
The voice recognition backend is currently hosted via Google Colab. Ensure the notebook is running for voice features to function correctly.

* **API Endpoint/Notebook:** [Google Colab Voice Recognition Drive](https://colab.research.google.com/drive/1UIDXW-KNdcVDCg6QKgBXkzciooL3Jbdl?usp=sharing)

---

## 🚀 Getting Started

To set up the project locally for development:

### 1. Prerequisites
Ensure you have Node.js installed.

### 2. Installation
Clone the repository and install the dependencies:

```bash
npm install
```

### 3. Development Server
Run the app in development mode:

```bash
npm start
```

Open http://localhost:3000 to view it in your browser. The page will reload when you make changes.

### 4. Building for Production
To build the app for deployment (generating the PWA service workers and manifest):

```bash
npm run build
```