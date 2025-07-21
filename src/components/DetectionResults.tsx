// import React from 'react';
// import { AlertTriangle, CheckCircle, XCircle, Eye, Clock, Shield } from 'lucide-react';

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

// interface DetectionResultsProps {
//   results: DetectionResult[];
// }

// const DetectionResults: React.FC<DetectionResultsProps> = ({ results }) => {
//   const getThreatColor = (level: string) => {
//     switch (level) {
//       case 'low': return 'text-green-400 bg-green-400/10 border-green-400/20';
//       case 'medium': return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20';
//       case 'high': return 'text-orange-400 bg-orange-400/10 border-orange-400/20';
//       case 'critical': return 'text-red-400 bg-red-400/10 border-red-400/20';
//       default: return 'text-gray-400 bg-gray-400/10 border-gray-400/20';
//     }
//   };

//   const getThreatIcon = (isDeepfake: boolean, level: string) => {
//     if (!isDeepfake) return <CheckCircle className="h-5 w-5 text-green-400" />;
//     if (level === 'critical') return <XCircle className="h-5 w-5 text-red-400" />;
//     return <AlertTriangle className="h-5 w-5 text-orange-400" />;
//   };

//   return (
//     <div className="space-y-6">
//       <div className="flex items-center justify-between">
//         <h3 className="text-2xl font-bold text-white">Detection Results</h3>
//         <div className="text-sm text-gray-400">
//           {results.length} file{results.length !== 1 ? 's' : ''} analyzed
//         </div>
//       </div>

//       <div className="grid gap-6">
//         {results.map((result) => (
//           <div key={result.id} className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
//             <div className="p-6">
//               {/* Header */}
//               <div className="flex items-start justify-between mb-6">
//                 <div className="flex items-center space-x-3">
//                   {getThreatIcon(result.isDeepfake, result.threatLevel)}
//                   <div>
//                     <h4 className="text-lg font-semibold text-white">{result.filename}</h4>
//                     <div className="flex items-center space-x-4 text-sm text-gray-400 mt-1">
//                       <span className="capitalize">{result.type}</span>
//                       <span className="flex items-center">
//                         <Clock className="h-3 w-3 mr-1" />
//                         {result.timestamp.toLocaleTimeString()}
//                       </span>
//                     </div>
//                   </div>
//                 </div>
                
//                 <div className={`px-3 py-1 rounded-full border text-xs font-medium ${getThreatColor(result.threatLevel)}`}>
//                   {result.threatLevel.toUpperCase()} RISK
//                 </div>
//               </div>

//               {/* Detection Status */}
//               <div className="mb-6">
//                 <div className="flex items-center justify-between mb-2">
//                   <span className="text-sm font-medium text-gray-300">Detection Confidence</span>
//                   <span className="text-sm font-bold text-white">{result.confidence.toFixed(1)}%</span>
//                 </div>
//                 <div className="w-full bg-gray-700 rounded-full h-2">
//                   <div
//                     className={`h-2 rounded-full transition-all duration-500 ${
//                       result.isDeepfake ? 'bg-gradient-to-r from-red-500 to-orange-500' : 'bg-gradient-to-r from-green-500 to-cyan-500'
//                     }`}
//                     style={{ width: `${result.confidence}%` }}
//                   ></div>
//                 </div>
//               </div>

//               {/* Analysis Details */}
//               <div className="grid md:grid-cols-3 gap-4 mb-6">
//                 <div className="bg-gray-700/50 rounded-lg p-4">
//                   <div className="flex items-center mb-2">
//                     <Eye className="h-4 w-4 text-cyan-400 mr-2" />
//                     <span className="text-sm font-medium text-gray-300">Face Detection</span>
//                   </div>
//                   <div className="text-lg font-bold text-white">{result.analysisDetails.faceDetection.toFixed(1)}%</div>
//                 </div>
                
//                 <div className="bg-gray-700/50 rounded-lg p-4">
//                   <div className="flex items-center mb-2">
//                     <Clock className="h-4 w-4 text-cyan-400 mr-2" />
//                     <span className="text-sm font-medium text-gray-300">Temporal Consistency</span>
//                   </div>
//                   <div className="text-lg font-bold text-white">{result.analysisDetails.temporalConsistency.toFixed(1)}%</div>
//                 </div>
                
//                 <div className="bg-gray-700/50 rounded-lg p-4">
//                   <div className="flex items-center mb-2">
//                     <Shield className="h-4 w-4 text-cyan-400 mr-2" />
//                     <span className="text-sm font-medium text-gray-300">Artifact Detection</span>
//                   </div>
//                   <div className="text-lg font-bold text-white">{result.analysisDetails.artifactDetection.toFixed(1)}%</div>
//                 </div>
//               </div>

//               {/* Verdict */}
//               <div className={`rounded-lg p-4 border ${
//                 result.isDeepfake 
//                   ? 'bg-red-900/20 border-red-600/30' 
//                   : 'bg-green-900/20 border-green-600/30'
//               }`}>
//                 <div className="flex items-center">
//                   {result.isDeepfake ? (
//                     <XCircle className="h-5 w-5 text-red-400 mr-3" />
//                   ) : (
//                     <CheckCircle className="h-5 w-5 text-green-400 mr-3" />
//                   )}
//                   <div>
//                     <h4 className={`font-semibold ${result.isDeepfake ? 'text-red-400' : 'text-green-400'}`}>
//                       {result.isDeepfake ? 'DEEPFAKE DETECTED' : 'AUTHENTIC MEDIA'}
//                     </h4>
//                     <p className="text-sm text-gray-300 mt-1">
//                       {result.isDeepfake 
//                         ? 'This media appears to contain artificially generated or manipulated content.'
//                         : 'This media appears to be authentic with no signs of manipulation.'
//                       }
//                     </p>
//                   </div>
//                 </div>
//               </div>
//             </div>
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// export default DetectionResults;

import React from 'react';
import {
  AlertTriangle,
  CheckCircle,
  XCircle,
  Eye,
  Clock,
  Shield
} from 'lucide-react';

interface DetectionResult {
  id: string;
  filename: string;
  type: 'image' | 'video';
  confidence: number;
  isDeepfake: boolean;
  threatLevel: 'low' | 'medium' | 'high' | 'critical';
  timestamp: string | Date;
  analysisDetails: {
    faceDetection: number;
    temporalConsistency: number;
    artifactDetection: number;
  };
}

interface DetectionResultsProps {
  results: DetectionResult[];
}

const DetectionResults: React.FC<DetectionResultsProps> = ({ results }) => {
  const getThreatColor = (level: string) => {
    switch (level) {
      case 'low':
        return 'text-green-400 bg-green-400/10 border-green-400/20';
      case 'medium':
        return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20';
      case 'high':
        return 'text-orange-400 bg-orange-400/10 border-orange-400/20';
      case 'critical':
        return 'text-red-400 bg-red-400/10 border-red-400/20';
      default:
        return 'text-gray-400 bg-gray-400/10 border-gray-400/20';
    }
  };

  const getThreatIcon = (isDeepfake: boolean, level: string) => {
    if (!isDeepfake)
      return <CheckCircle className="h-5 w-5 text-green-400" />;
    if (level === 'critical')
      return <XCircle className="h-5 w-5 text-red-400" />;
    return <AlertTriangle className="h-5 w-5 text-orange-400" />;
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-2xl font-bold text-white">Detection Results</h3>
        <div className="text-sm text-gray-400">
          {results.length} file{results.length !== 1 ? 's' : ''} analyzed
        </div>
      </div>

      <div className="grid gap-6">
        {results.map((result) => (
          <div
            key={result.id}
            className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden"
          >
            <div className="p-6">
              {/* Header */}
              <div className="flex items-start justify-between mb-6">
                <div className="flex items-center space-x-3">
                  {getThreatIcon(result.isDeepfake, result.threatLevel)}
                  <div>
                    <h4 className="text-lg font-semibold text-white">
                      {result.filename}
                    </h4>
                    <div className="flex items-center space-x-4 text-sm text-gray-400 mt-1">
                      <span className="capitalize">{result.type}</span>
                      <span className="flex items-center">
                        <Clock className="h-3 w-3 mr-1" />
                        {new Date(result.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                </div>

                <div
                  className={`px-3 py-1 rounded-full border text-xs font-medium ${getThreatColor(
                    result.threatLevel
                  )}`}
                >
                  {result.threatLevel.toUpperCase()} RISK
                </div>
              </div>

              {/* Detection Status */}
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-300">
                    Detection Confidence
                  </span>
                  <span className="text-sm font-bold text-white">
                    {result.confidence.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-500 ${
                      result.isDeepfake
                        ? 'bg-gradient-to-r from-red-500 to-orange-500'
                        : 'bg-gradient-to-r from-green-500 to-cyan-500'
                    }`}
                    style={{ width: `${result.confidence}%` }}
                  ></div>
                </div>
              </div>

              {/* Analysis Details */}
              <div className="grid md:grid-cols-3 gap-4 mb-6">
                <div className="bg-gray-700/50 rounded-lg p-4">
                  <div className="flex items-center mb-2">
                    <Eye className="h-4 w-4 text-cyan-400 mr-2" />
                    <span className="text-sm font-medium text-gray-300">
                      Face Detection
                    </span>
                  </div>
                  <div className="text-lg font-bold text-white">
                    {result.analysisDetails.faceDetection.toFixed(1)}%
                  </div>
                </div>

                <div className="bg-gray-700/50 rounded-lg p-4">
                  <div className="flex items-center mb-2">
                    <Clock className="h-4 w-4 text-cyan-400 mr-2" />
                    <span className="text-sm font-medium text-gray-300">
                      Temporal Consistency
                    </span>
                  </div>
                  <div className="text-lg font-bold text-white">
                    {result.analysisDetails.temporalConsistency.toFixed(1)}%
                  </div>
                </div>

                <div className="bg-gray-700/50 rounded-lg p-4">
                  <div className="flex items-center mb-2">
                    <Shield className="h-4 w-4 text-cyan-400 mr-2" />
                    <span className="text-sm font-medium text-gray-300">
                      Artifact Detection
                    </span>
                  </div>
                  <div className="text-lg font-bold text-white">
                    {result.analysisDetails.artifactDetection.toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Verdict */}
              <div
                className={`rounded-lg p-4 border ${
                  result.isDeepfake
                    ? 'bg-red-900/20 border-red-600/30'
                    : 'bg-green-900/20 border-green-600/30'
                }`}
              >
                <div className="flex items-center">
                  {result.isDeepfake ? (
                    <XCircle className="h-5 w-5 text-red-400 mr-3" />
                  ) : (
                    <CheckCircle className="h-5 w-5 text-green-400 mr-3" />
                  )}
                  <div>
                    <h4
                      className={`font-semibold ${
                        result.isDeepfake
                          ? 'text-red-400'
                          : 'text-green-400'
                      }`}
                    >
                      {result.isDeepfake
                        ? 'DEEPFAKE DETECTED'
                        : 'AUTHENTIC MEDIA'}
                    </h4>
                    <p className="text-sm text-gray-300 mt-1">
                      {result.isDeepfake
                        ? 'This media appears to contain artificially generated or manipulated content.'
                        : 'This media appears to be authentic with no signs of manipulation.'}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DetectionResults;
