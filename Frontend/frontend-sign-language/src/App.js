import React, { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import './App.css';

function App() {
  const webcamRef = useRef(null);
  const [letter, setLetter] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [word, setWord] = useState('');

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: 'user'
  };

  const roi = { x: 220, y: 140, width: 200, height: 200 };

  const capture = () => {
    const screenshot = webcamRef.current.getScreenshot();
    if (!screenshot) return;

    const image = new Image();
    image.src = screenshot;
    image.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = roi.width;
      canvas.height = roi.height;
      const ctx = canvas.getContext('2d');

      ctx.drawImage(
        image,
        roi.x,
        roi.y,
        roi.width,
        roi.height,
        0,
        0,
        roi.width,
        roi.height
      );

      canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('file', blob, 'image.jpg');

        axios.post('http://127.0.0.1:5000/predict', formData)
          .then(res => {
            setLetter(res.data.letter);
            setConfidence(res.data.confidence);
          })
          .catch(err => {
            console.error('Error al predecir:', err);
          });
      }, 'image/jpeg');
    };
  };

  useEffect(() => {
    const interval = setInterval(() => {
      capture();
    }, 100); // Ejecutar cada 100 ms
    return () => clearInterval(interval);
  }, []);

const saveLetter = () => {
  console.log('Letra:', letter, 'Confianza:', confidence);  // Depuración

  if (letter && confidence > 0.01) {
    setWord(prev => prev + letter);
  } else {
  }
};


  const deleteLetter = () => {
    setWord(prev => prev.slice(0, -1));
  };

  const clearWord = () => {
    setWord('');
  };

  const sendWord = () => {
    if (word) {
      axios.post('http://127.0.0.1:5000/execute', { word })
        .then(res => {
          alert(res.data.message || 'Comando ejecutado');
        })
        .catch(err => {
          console.error('Error al enviar la palabra:', err);
        });
    }
  };

  return (
    <div className="App">
      <h1>Traductor de Señas</h1>
      <div className="webcam-container" style={{ position: 'relative' }}>
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
          className="webcam-feed"
        />
        <div
          style={{
            position: 'absolute',
            border: '3px solid green',
            left: `${roi.x}px`,
            top: `${roi.y}px`,
            width: `${roi.width}px`,
            height: `${roi.height}px`,
            pointerEvents: 'none'
          }}
        ></div>
      </div>
      <div className="buttons">
        <button onClick={saveLetter}>📸 Guardar letra</button>
        <button onClick={deleteLetter}>Borrar Letra</button>
        <button onClick={clearWord}>Limpiar Palabra</button>
        <button onClick={sendWord}>Enviar palabra</button>
      </div>
      <h2>
        Letra detectada:{' '}
        <span style={{ color: 'green' }}>
          {letter ? `${letter} (${(confidence * 100).toFixed(1)}%)` : '(0.0%)'}
        </span>
      </h2>
      <p><strong>Palabra:</strong> {word}</p>
    </div>
  );
}

export default App;
