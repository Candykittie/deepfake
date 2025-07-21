// import React, { useState, useEffect } from 'react';
// import { Upload, Shield, AlertTriangle, Eye, Activity, Database, FileText, Download, Brain, TrendingUp, Layers } from 'lucide-react';
// import UploadZone from './components/UploadZone';
// import MLModelStatus from './components/MLModelStatus';
// import DetectionResults from './components/DetectionResults';
// import Dashboard from './components/Dashboard';
// import ThreatMonitor from './components/ThreatMonitor';
// import AuditLog from './components/AuditLog';
// import AGIAssistant from './components/AGIAssistant';
// import PredictiveAnalytics from './components/PredictiveAnalytics';
// import NeuralNetworkVisualizer from './components/NeuralNetworkVisualizer';
// import { DeepfakeDetector } from './utils/deepfakeDetector';

// interface DetectionResult {
//   id: string;
//   filename: string;
//   type: 'image' | 'video';
//   confidence: number;
//   isDeepfake: boolean;
//   threatLevel: 'low' | 'medium' | 'high' | 'critical';
//   timestamp: Date;
//   analysisDetails: {
//     faceDetection: number;
//     temporalConsistency: number;
//     artifactDetection: number;
//   };
// }

// function App() {
//   const [activeTab, setActiveTab] = useState<'upload' | 'dashboard' | 'monitor' | 'audit' | 'agi' | 'predictive' | 'neural'>('upload');
//   const [detectionResults, setDetectionResults] = useState<DetectionResult[]>([]);
//   const [isProcessing, setIsProcessing] = useState(false);
//   const [isModelReady, setIsModelReady] = useState(false);
//   const [areDetectorModelsReady, setAreDetectorModelsReady] = useState(false);
//   const [detector] = useState(() => new DeepfakeDetector());

//   useEffect(() => {
//     const initializeDetector = async () => {
//       if (isModelReady && !areDetectorModelsReady) {
//         try {
//           await detector.initialize();
//           setAreDetectorModelsReady(true);
//         } catch (error) {
//           console.error('Failed to initialize detector:', error);
//         }
//       }
//     };

//     initializeDetector();
//   }, [isModelReady, areDetectorModelsReady, detector]);

//   const handleFileUpload = async (files: File[]) => {
//     if (!areDetectorModelsReady) {
//       alert('Please wait for ML models to finish loading');
//       return;
//     }

//     setIsProcessing(true);
    
//     // Process each file with real ML detection
//     for (const file of files) {
//       try {
//         const mlResult = await detector.detectDeepfake(file);
        
//         const result: DetectionResult = {
//           id: Math.random().toString(36).substr(2, 9),
//           filename: file.name,
//           type: file.type.startsWith('video/') ? 'video' : 'image',
//           confidence: mlResult.confidence,
//           isDeepfake: mlResult.isDeepfake,
//           threatLevel: mlResult.threatLevel,
//           timestamp: new Date(),
//           analysisDetails: {
//             faceDetection: mlResult.analysis.faceDetection,
//             temporalConsistency: mlResult.analysis.temporalConsistency,
//             artifactDetection: mlResult.analysis.artifactDetection,
//           }
//         };
        
//         setDetectionResults(prev => [result, ...prev]);
//       } 
//       catch (error) {
//         // console.error('Detection failed for file:', file.name, error);
        
//         // Fallback result in case of error
//         const fallbackResult: DetectionResult = {
//           id: Math.random().toString(36).substr(2, 9),
//           filename: file.name,
//           type: file.type.startsWith('video/') ? 'video' : 'image',
//           confidence: 0,
//           isDeepfake: false,
//           threatLevel: 'low',
//           timestamp: new Date(),
//           analysisDetails: {
//             faceDetection: 0,
//             temporalConsistency: 0,
//             artifactDetection: 0,
//           }
//         };
        
//         setDetectionResults(prev => [fallbackResult, ...prev]);
//       }
//     }
    
//     setIsProcessing(false);
//   };

//   const handleThreatAction = (action: string, threatId: string) => {
//     console.log(`AGI Action: ${action} on threat ${threatId}`);
//     // Implement threat action handling
//   };

//   const navItems = [
//     { id: 'upload', label: 'Detection', icon: Upload },
//     { id: 'dashboard', label: 'Dashboard', icon: Activity },
//     { id: 'monitor', label: 'Threat Monitor', icon: Eye },
//     { id: 'audit', label: 'Audit Log', icon: Database },
//     { id: 'agi', label: 'AGI Assistant', icon: Brain },
//     { id: 'predictive', label: 'Predictive Analytics', icon: TrendingUp },
//     { id: 'neural', label: 'Neural Network', icon: Layers },
//   ];

//   return (
//     <div className="min-h-screen bg-gray-900 text-white">
//       {/* Header */}
//       <header className="bg-gray-800 border-b border-gray-700">
//         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
//           <div className="flex items-center justify-between h-16">
//             <div className="flex items-center">
//               <Shield className="h-8 w-8 text-cyan-400 mr-3" />
//               <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
//                 DeepGuard AI
//               </h1>
//             </div>
//             <div className="flex items-center space-x-4">
//               <div className="flex items-center text-sm text-gray-300">
//                 <div className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></div>
//                 System Online
//               </div>
//             </div>
//           </div>
//         </div>
//       </header>

//       {/* Navigation */}
//       <nav className="bg-gray-800 border-b border-gray-700">
//         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
//           <div className="flex space-x-8">
//             {navItems.map((item) => {
//               const Icon = item.icon;
//               return (
//                 <button
//                   key={item.id}
//                   onClick={() => setActiveTab(item.id as any)}
//                   className={`flex items-center px-3 py-4 text-sm font-medium border-b-2 transition-colors ${
//                     activeTab === item.id
//                       ? 'border-cyan-400 text-cyan-400'
//                       : 'border-transparent text-gray-300 hover:text-white hover:border-gray-300'
//                   }`}
//                 >
//                   <Icon className="h-4 w-4 mr-2" />
//                   {item.label}
//                 </button>
//               );
//             })}
//           </div>
//         </div>
//       </nav>

//       {/* Main Content */}
//       <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
//         {activeTab === 'upload' && (
//           <div className="space-y-8">
//             <div>
//               <h2 className="text-3xl font-bold text-white mb-2">Deepfake Detection</h2>
//               <p className="text-gray-400">Upload images or videos for AI-powered deepfake detection using advanced CNN models</p>
//             </div>
            
//             <MLModelStatus onModelReady={setIsModelReady} />
//             <UploadZone onFilesUpload={handleFileUpload} isProcessing={isProcessing} isModelReady={areDetectorModelsReady} />
            
//             {detectionResults.length > 0 && (
//               <DetectionResults results={detectionResults} />
//             )}
//           </div>
//         )}

//         {activeTab === 'dashboard' && <Dashboard results={detectionResults} />}
//         {activeTab === 'monitor' && <ThreatMonitor results={detectionResults} />}
//         {activeTab === 'audit' && <AuditLog results={detectionResults} />}
//         {activeTab === 'agi' && <AGIAssistant results={detectionResults} onThreatAction={handleThreatAction} />}
//         {activeTab === 'predictive' && <PredictiveAnalytics results={detectionResults} />}
//         {activeTab === 'neural' && <NeuralNetworkVisualizer isProcessing={isProcessing} detectionResults={detectionResults} />}
//       </main>
//     </div>
//   );
// }

// export default App;




import React, { useState, useEffect } from 'react';
import {
  Upload, Shield, Eye, Activity, Database,
  Brain, TrendingUp, Layers
} from 'lucide-react';

import UploadZone from './components/UploadZone';
import MLModelStatus from './components/MLModelStatus';
import DetectionResults from './components/DetectionResults';
import Dashboard from './components/Dashboard';
import ThreatMonitor from './components/ThreatMonitor';
import AuditLog from './components/AuditLog';
import AGIAssistant from './components/AGIAssistant';
import PredictiveAnalytics from './components/PredictiveAnalytics';
import NeuralNetworkVisualizer from './components/NeuralNetworkVisualizer';
import { DeepfakeDetector } from './utils/deepfakeDetector';

interface DetectionResult {
  id: string;
  filename: string;
  type: 'image' | 'video';
  confidence: number;
  isDeepfake: boolean;
  threatLevel: 'low' | 'medium' | 'high' | 'critical';
  timestamp: Date;
  analysisDetails: {
    faceDetection: number;
    temporalConsistency: number;
    artifactDetection: number;
  };
}

function App() {
  const [activeTab, setActiveTab] = useState<'upload' | 'dashboard' | 'monitor' | 'audit' | 'agi' | 'predictive' | 'neural'>('upload');
  const [detectionResults, setDetectionResults] = useState<DetectionResult[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isModelReady, setIsModelReady] = useState(false);
  const [areDetectorModelsReady, setAreDetectorModelsReady] = useState(false);
  const [detector] = useState(() => new DeepfakeDetector());

  useEffect(() => {
    const initializeDetector = async () => {
      if (isModelReady && !areDetectorModelsReady) {
        try {
          await detector.initialize();
          setAreDetectorModelsReady(true);
        } catch (error) {
          console.error('Failed to initialize detector:', error);
        }
      }
    };
    initializeDetector();
  }, [isModelReady, areDetectorModelsReady, detector]);

  const handleFileUpload = async (files: File[]) => {
    if (!areDetectorModelsReady) {
      alert('Please wait for ML models to finish loading');
      return;
    }

    setIsProcessing(true);

    for (const file of files) {
      try {
        const mlResult = await detector.detectDeepfake(file);
        console.log('Detection Result:', mlResult);

        const result: DetectionResult = {
          id: Math.random().toString(36).substr(2, 9),
          filename: file.name,
          type: file.type.startsWith('video/') ? 'video' : 'image',
          confidence: mlResult.confidence,
          isDeepfake: mlResult.isDeepfake,
          threatLevel: mlResult.threatLevel,
          timestamp: new Date(),
          analysisDetails: {
            faceDetection: mlResult.analysis.faceDetection,
            temporalConsistency: mlResult.analysis.temporalConsistency,
            artifactDetection: mlResult.analysis.artifactDetection,
          }
        };

        setDetectionResults(prev => [result, ...prev]);
      } catch (error) {
        const fallbackResult: DetectionResult = {
          id: Math.random().toString(36).substr(2, 9),
          filename: file.name,
          type: file.type.startsWith('video/') ? 'video' : 'image',
          confidence: 0,
          isDeepfake: false,
          threatLevel: 'low',
          timestamp: new Date(),
          analysisDetails: {
            faceDetection: 0,
            temporalConsistency: 0,
            artifactDetection: 0,
          }
        };
        setDetectionResults(prev => [fallbackResult, ...prev]);
      }
    }

    setIsProcessing(false);
  };

  const handleThreatAction = (action: string, threatId: string) => {
    console.log(`AGI Action: ${action} on threat ${threatId}`);
  };

  const navItems = [
    { id: 'upload', label: 'Detection', icon: Upload },
    { id: 'dashboard', label: 'Dashboard', icon: Activity },
    { id: 'monitor', label: 'Threat Monitor', icon: Eye },
    { id: 'audit', label: 'Audit Log', icon: Database },
    { id: 'agi', label: 'AGI Assistant', icon: Brain },
    { id: 'predictive', label: 'Predictive Analytics', icon: TrendingUp },
    { id: 'neural', label: 'Neural Network', icon: Layers },
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <Shield className="h-8 w-8 text-cyan-400 mr-3" />
              <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                DeepGuard AI
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center text-sm text-gray-300">
                <div className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></div>
                System Online
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id as any)}
                  className={`flex items-center px-3 py-4 text-sm font-medium border-b-2 transition-colors ${
                    activeTab === item.id
                      ? 'border-cyan-400 text-cyan-400'
                      : 'border-transparent text-gray-300 hover:text-white hover:border-gray-300'
                  }`}
                >
                  <Icon className="h-4 w-4 mr-2" />
                  {item.label}
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'upload' && (
          <div className="space-y-8">
            <div>
              <h2 className="text-3xl font-bold text-white mb-2">Deepfake Detection</h2>
              <p className="text-gray-400">Upload images or videos for AI-powered deepfake detection using advanced CNN models</p>
            </div>

            <MLModelStatus onModelReady={setIsModelReady} />
            <UploadZone onFilesUpload={handleFileUpload} isProcessing={isProcessing} isModelReady={areDetectorModelsReady} />

            {detectionResults.length > 0 && (
              <DetectionResults results={detectionResults} />
            )}
          </div>
        )}

        {activeTab === 'dashboard' && <Dashboard results={detectionResults} />}
        {activeTab === 'monitor' && <ThreatMonitor results={detectionResults} />}
        {activeTab === 'audit' && <AuditLog results={detectionResults} />}
        {activeTab === 'agi' && <AGIAssistant results={detectionResults} onThreatAction={handleThreatAction} />}
        {activeTab === 'predictive' && <PredictiveAnalytics results={detectionResults} />}
        {activeTab === 'neural' && <NeuralNetworkVisualizer isProcessing={isProcessing} detectionResults={detectionResults} />}
      </main>
    </div>
  );
}

export default App;
