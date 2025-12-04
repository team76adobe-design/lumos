import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import Chat from './Chat';
import {
  BrowserRouter as Router,
  Routes,
  Route
} from 'react-router-dom';
import reportWebVitals from './reportWebVitals';
import Edit from './Edit.js'
import GLBModel from './3DPage.jsx'
import { PageTransition, AnimatedPageContainer } from './PageTransition';
import Discover from './Discover.tsx';
import { Toaster } from 'sonner';
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    {/* <PasswordGate> */}
      <Router>
        <Routes>
          <Route path="/" element={<PageTransition><App /></PageTransition>} />
          <Route path="/discover" element={<Discover />} />
          <Route path="/ai" element={<PageTransition><AnimatedPageContainer><Chat /></AnimatedPageContainer></PageTransition>} />
          <Route path="/photoroom" element={<PageTransition><AnimatedPageContainer><Edit /></AnimatedPageContainer></PageTransition>} />
          <Route path="/3d" element={<PageTransition><AnimatedPageContainer><GLBModel /></AnimatedPageContainer></PageTransition>} />
        </Routes>
      </Router>
      <Toaster
        position="top-center"
        richColors
        theme="dark"
        toastOptions={{
          style: {
            background: '#1f1f1f',
            color: '#fff',
            border: '1px solid rgba(255,255,255,0.1)',
          },
          className: 'sonner-toast',
        }}
      />
    {/* </PasswordGate> */}
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();

// Register a minimal service worker so the site is recognized as a PWA
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker
      .register('/sw.js')
      .then((registration) => {
        console.log('ServiceWorker registration successful with scope: ', registration.scope);
      })
      .catch((err) => {
        console.warn('ServiceWorker registration failed: ', err);
      });
  });
}
