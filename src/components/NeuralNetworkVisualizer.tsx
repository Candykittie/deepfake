import React, { useState, useEffect, useRef } from 'react';
import { Brain, Layers, Zap, Activity, Settings, Play, Pause } from 'lucide-react';

interface NeuralNetworkVisualizerProps {
  isProcessing: boolean;
  detectionResults: any[];
}

interface Neuron {
  id: string;
  x: number;
  y: number;
  activation: number;
  layer: number;
}

interface Connection {
  from: string;
  to: string;
  weight: number;
  active: boolean;
}

const NeuralNetworkVisualizer: React.FC<NeuralNetworkVisualizerProps> = ({ 
  isProcessing, 
  detectionResults 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [neurons, setNeurons] = useState<Neuron[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [isAnimating, setIsAnimating] = useState(false);
  const [networkStats, setNetworkStats] = useState({
    layers: 8,
    neurons: 256,
    connections: 1024,
    accuracy: 94.7,
    learningRate: 0.001
  });

  useEffect(() => {
    initializeNetwork();
  }, []);

  useEffect(() => {
    if (isProcessing) {
      setIsAnimating(true);
      animateNetwork();
    }
  }, [isProcessing]);

  const initializeNetwork = () => {
    const newNeurons: Neuron[] = [];
    const newConnections: Connection[] = [];
    
    // Create layers with different neuron counts
    const layerSizes = [64, 128, 256, 512, 256, 128, 64, 2]; // Input to output
    const layerSpacing = 100;
    const neuronSpacing = 30;
    
    let neuronId = 0;
    
    layerSizes.forEach((size, layerIndex) => {
      const startY = (400 - (size * neuronSpacing)) / 2;
      
      for (let i = 0; i < Math.min(size, 12); i++) { // Limit visual neurons
        newNeurons.push({
          id: `neuron-${neuronId++}`,
          x: layerIndex * layerSpacing + 50,
          y: startY + i * neuronSpacing,
          activation: Math.random(),
          layer: layerIndex
        });
      }
    });

    // Create connections between adjacent layers
    for (let layer = 0; layer < layerSizes.length - 1; layer++) {
      const currentLayerNeurons = newNeurons.filter(n => n.layer === layer);
      const nextLayerNeurons = newNeurons.filter(n => n.layer === layer + 1);
      
      currentLayerNeurons.forEach(fromNeuron => {
        nextLayerNeurons.forEach(toNeuron => {
          if (Math.random() > 0.7) { // Sparse connections for visualization
            newConnections.push({
              from: fromNeuron.id,
              to: toNeuron.id,
              weight: (Math.random() - 0.5) * 2,
              active: false
            });
          }
        });
      });
    }

    setNeurons(newNeurons);
    setConnections(newConnections);
  };

  const animateNetwork = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrame = 0;
    const maxFrames = 120;

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw connections
      connections.forEach(conn => {
        const fromNeuron = neurons.find(n => n.id === conn.from);
        const toNeuron = neurons.find(n => n.id === conn.to);
        
        if (fromNeuron && toNeuron) {
          const progress = animationFrame / maxFrames;
          const isActive = Math.sin(progress * Math.PI * 4 + fromNeuron.layer) > 0.5;
          
          ctx.strokeStyle = isActive 
            ? `rgba(34, 211, 238, ${Math.abs(conn.weight) * 0.8})` 
            : `rgba(75, 85, 99, 0.3)`;
          ctx.lineWidth = isActive ? 2 : 1;
          
          ctx.beginPath();
          ctx.moveTo(fromNeuron.x, fromNeuron.y);
          ctx.lineTo(toNeuron.x, toNeuron.y);
          ctx.stroke();
        }
      });

      // Draw neurons
      neurons.forEach(neuron => {
        const progress = animationFrame / maxFrames;
        const activation = Math.sin(progress * Math.PI * 2 + neuron.layer * 0.5) * 0.5 + 0.5;
        
        // Neuron body
        ctx.fillStyle = `rgba(34, 211, 238, ${activation * 0.8 + 0.2})`;
        ctx.beginPath();
        ctx.arc(neuron.x, neuron.y, 8, 0, Math.PI * 2);
        ctx.fill();
        
        // Neuron glow
        if (activation > 0.7) {
          ctx.shadowColor = '#22d3ee';
          ctx.shadowBlur = 15;
          ctx.fillStyle = `rgba(34, 211, 238, ${activation})`;
          ctx.beginPath();
          ctx.arc(neuron.x, neuron.y, 12, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
        }
      });

      animationFrame++;
      if (animationFrame < maxFrames && isAnimating) {
        requestAnimationFrame(animate);
      } else {
        setIsAnimating(false);
      }
    };

    animate();
  };

  const toggleAnimation = () => {
    setIsAnimating(!isAnimating);
    if (!isAnimating) {
      animateNetwork();
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-white mb-2">Neural Network Visualizer</h2>
          <p className="text-gray-400">Real-time visualization of deep learning model architecture and processing</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <button
            onClick={toggleAnimation}
            className="flex items-center px-4 py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 transition-colors"
          >
            {isAnimating ? <Pause className="h-4 w-4 mr-2" /> : <Play className="h-4 w-4 mr-2" />}
            {isAnimating ? 'Pause' : 'Animate'}
          </button>
        </div>
      </div>

      {/* Network Stats */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center">
            <Layers className="h-6 w-6 text-cyan-400 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-400">Layers</p>
              <p className="text-xl font-bold text-white">{networkStats.layers}</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center">
            <Brain className="h-6 w-6 text-purple-400 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-400">Neurons</p>
              <p className="text-xl font-bold text-white">{networkStats.neurons}</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center">
            <Zap className="h-6 w-6 text-yellow-400 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-400">Connections</p>
              <p className="text-xl font-bold text-white">{networkStats.connections}</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center">
            <Activity className="h-6 w-6 text-green-400 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-400">Accuracy</p>
              <p className="text-xl font-bold text-white">{networkStats.accuracy}%</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center">
            <Settings className="h-6 w-6 text-orange-400 mr-3" />
            <div>
              <p className="text-sm font-medium text-gray-400">Learning Rate</p>
              <p className="text-xl font-bold text-white">{networkStats.learningRate}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Network Visualization */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
        <div className="bg-gray-700 px-6 py-4 border-b border-gray-600">
          <h3 className="text-lg font-semibold text-white">Deep Learning Architecture</h3>
          <p className="text-sm text-gray-400 mt-1">
            Convolutional Neural Network with attention mechanisms for deepfake detection
          </p>
        </div>
        
        <div className="p-6">
          <canvas
            ref={canvasRef}
            width={800}
            height={400}
            className="w-full h-auto bg-gray-900 rounded-lg border border-gray-600"
          />
        </div>

        {/* Layer Information */}
        <div className="px-6 pb-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gray-700/50 rounded-lg p-4">
              <h4 className="text-sm font-semibold text-cyan-400 mb-2">Input Layer</h4>
              <p className="text-xs text-gray-300">
                Raw image/video data preprocessing and feature extraction
              </p>
            </div>
            
            <div className="bg-gray-700/50 rounded-lg p-4">
              <h4 className="text-sm font-semibold text-purple-400 mb-2">Convolutional Layers</h4>
              <p className="text-xs text-gray-300">
                Spatial feature detection and pattern recognition
              </p>
            </div>
            
            <div className="bg-gray-700/50 rounded-lg p-4">
              <h4 className="text-sm font-semibold text-yellow-400 mb-2">Attention Mechanism</h4>
              <p className="text-xs text-gray-300">
                Focus on suspicious regions and temporal inconsistencies
              </p>
            </div>
            
            <div className="bg-gray-700/50 rounded-lg p-4">
              <h4 className="text-sm font-semibold text-green-400 mb-2">Output Layer</h4>
              <p className="text-xs text-gray-300">
                Binary classification: authentic vs. deepfake
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Processing Status */}
      {isProcessing && (
        <div className="bg-gradient-to-r from-cyan-900/20 to-blue-900/20 border border-cyan-600/30 rounded-lg p-6">
          <div className="flex items-center space-x-4">
            <Brain className="h-8 w-8 text-cyan-400 animate-pulse" />
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-cyan-400">Neural Network Processing</h3>
              <p className="text-gray-300">Forward propagation through {networkStats.layers} layers...</p>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-400">Processing Speed</p>
              <p className="text-xl font-bold text-white">847 FPS</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default NeuralNetworkVisualizer;