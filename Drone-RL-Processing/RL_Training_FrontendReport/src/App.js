import logo from './logo.svg';
import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './dash.js'

function App() {
  return (
    <div className="App">
      <Router>
      <Routes>
      <Route path="/" element={<Dashboard />} />
      </Routes>
      </Router>
    </div>
  );
}

export default App;
