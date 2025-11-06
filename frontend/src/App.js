import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import HomePage from './pages/HomePage';
import SugyeoPage from './pages/SugyeoPage';

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/sugyeo" element={<SugyeoPage />} />
      </Routes>
    </Layout>
  );
}

export default App; 