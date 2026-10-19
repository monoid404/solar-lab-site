import React from 'react';
import { createRoot } from 'react-dom/client';
import SolarLabApp from './App.jsx';
import './style.css';

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <SolarLabApp />
  </React.StrictMode>,
);
