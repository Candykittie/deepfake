import React from 'react';
import { Shield, AlertTriangle, CheckCircle, TrendingUp, Activity, FileText, Cpu, Zap, Gauge } from 'lucide-react';

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

interface DashboardProps {
  results: DetectionResult[];
}

const Dashboard: React.FC<DashboardProps> = ({ results }) => {
  const [processorStats, setProcessorStats] = React.useState({
    cpuUsage: 0,
    processingSpeed: 0,
    performanceScore: 0,
    memoryUsage: 0,
    threadsActive: 0
  });

  React.useEffect(() => {
    const updateProcessorStats = () => {
      // Simulate real-time processor metrics
      const baseSpeed = 847 + Math.sin(Date.now() / 1000) * 50;
      const cpuLoad = 35 + Math.random() * 30;
      const memUsage = 45 + Math.random() * 25;
      const threads = Math.floor(4 + Math.random() * 4);
      
      // Calculate performance score based on various factors
      const speedFactor = Math.min(baseSpeed / 1000, 1);
      const cpuFactor = Math.max(0, (100 - cpuLoad) / 100);
      const memFactor = Math.max(0, (100 - memUsage) / 100);
      const performanceScore = (speedFactor * 0.4 + cpuFactor * 0.3 + memFactor * 0.3) * 100;
      
      setProcessorStats({
        cpuUsage: cpuLoad,
        processingSpeed: baseSpeed,
        performanceScore: performanceScore,
        memoryUsage: memUsage,
        threadsActive: threads
      });
    };

    updateProcessorStats();
    const interval = setInterval(updateProcessorStats, 2000);
    return () => clearInterval(interval);
  }, []);

  const totalFiles = results.length;
  const deepfakeCount = results.filter(r => r.isDeepfake).length;
  const authenticCount = totalFiles - deepfakeCount;
  const criticalThreats = results.filter(r => r.threatLevel === 'critical').length;
  const averageConfidence = totalFiles > 0 ? results.reduce((acc, r) => acc + r.confidence, 0) / totalFiles : 0;

  const threatDistribution = {
    low: results.filter(r => r.threatLevel === 'low').length,
    medium: results.filter(r => r.threatLevel === 'medium').length,
    high: results.filter(r => r.threatLevel === 'high').length,
    critical: results.filter(r => r.threatLevel === 'critical').length,
  };

  const recentActivity = results.slice(0, 5);

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-white mb-2">Security Dashboard</h2>
        <p className="text-gray-400">Real-time deepfake detection analytics and threat monitoring</p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <FileText className="h-8 w-8 text-cyan-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-400">Total Analyzed</p>
              <p className="text-2xl font-bold text-white">{totalFiles}</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <AlertTriangle className="h-8 w-8 text-red-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-400">Deepfakes Detected</p>
              <p className="text-2xl font-bold text-white">{deepfakeCount}</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <CheckCircle className="h-8 w-8 text-green-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-400">Authentic Media</p>
              <p className="text-2xl font-bold text-white">{authenticCount}</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <TrendingUp className="h-8 w-8 text-orange-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-400">Avg. Confidence</p>
              <p className="text-2xl font-bold text-white">{averageConfidence.toFixed(1)}%</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <Zap className="h-8 w-8 text-yellow-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-400">Processing Speed</p>
              <p className="text-2xl font-bold text-white">{processorStats.processingSpeed.toFixed(0)} FPS</p>
            </div>
          </div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Threat Level Distribution */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-6">Threat Level Distribution</h3>
          <div className="space-y-4">
            {Object.entries(threatDistribution).map(([level, count]) => {
              const percentage = totalFiles > 0 ? (count / totalFiles) * 100 : 0;
              const colors = {
                low: 'bg-green-500',
                medium: 'bg-yellow-500',
                high: 'bg-orange-500',
                critical: 'bg-red-500'
              };
              
              return (
                <div key={level}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-300 capitalize">{level} Risk</span>
                    <span className="text-white font-medium">{count}</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${colors[level as keyof typeof colors]} transition-all duration-500`}
                      style={{ width: `${percentage}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Detection Accuracy */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-6">System Performance</h3>
          <div className="space-y-6">
            <div className="text-center">
              <div className="relative inline-flex items-center justify-center w-24 h-24 mb-4">
                <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
                  <circle
                    cx="50"
                    cy="50"
                    r="40"
                    stroke="currentColor"
                    strokeWidth="8"
                    fill="none"
                    className="text-gray-700"
                  />
                  <circle
                    cx="50"
                    cy="50"
                    r="40"
                    stroke="currentColor"
                    strokeWidth="8"
                    fill="none"
                    strokeDasharray={`${averageConfidence * 2.51} 251`}
                    className="text-cyan-400"
                  />
                </svg>
                <span className="absolute text-xl font-bold text-white">
                  {averageConfidence.toFixed(0)}%
                </span>
              </div>
              <p className="text-sm text-gray-400">Average Detection Accuracy</p>
            </div>
            
            <div className="grid grid-cols-2 gap-4 text-center">
              <div>
                <p className="text-2xl font-bold text-green-400">{authenticCount}</p>
                <p className="text-xs text-gray-400">Authentic</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-red-400">{deepfakeCount}</p>
                <p className="text-xs text-gray-400">Deepfakes</p>
              </div>
            </div>
          </div>
        </div>

        {/* Processor Performance */}
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-6">Processor Performance</h3>
          <div className="space-y-6">
            {/* Performance Score Circle */}
            <div className="text-center">
              <div className="relative inline-flex items-center justify-center w-24 h-24 mb-4">
                <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
                  <circle
                    cx="50"
                    cy="50"
                    r="40"
                    stroke="currentColor"
                    strokeWidth="8"
                    fill="none"
                    className="text-gray-700"
                  />
                  <circle
                    cx="50"
                    cy="50"
                    r="40"
                    stroke="currentColor"
                    strokeWidth="8"
                    fill="none"
                    strokeDasharray={`${processorStats.performanceScore * 2.51} 251`}
                    className={`transition-all duration-1000 ${
                      processorStats.performanceScore > 80 ? 'text-green-400' :
                      processorStats.performanceScore > 60 ? 'text-yellow-400' : 'text-red-400'
                    }`}
                  />
                </svg>
                <span className="absolute text-lg font-bold text-white">
                  {processorStats.performanceScore.toFixed(0)}
                </span>
              </div>
              <p className="text-sm text-gray-400">Performance Score</p>
            </div>
            
            {/* Processor Metrics */}
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-400">CPU Usage</span>
                  <span className="text-white">{processorStats.cpuUsage.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-1000 ${
                      processorStats.cpuUsage > 80 ? 'bg-red-500' :
                      processorStats.cpuUsage > 60 ? 'bg-yellow-500' : 'bg-green-500'
                    }`}
                    style={{ width: `${processorStats.cpuUsage}%` }}
                  ></div>
                </div>
              </div>
              
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-400">Memory Usage</span>
                  <span className="text-white">{processorStats.memoryUsage.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-1000 ${
                      processorStats.memoryUsage > 80 ? 'bg-red-500' :
                      processorStats.memoryUsage > 60 ? 'bg-yellow-500' : 'bg-green-500'
                    }`}
                    style={{ width: `${processorStats.memoryUsage}%` }}
                  ></div>
                </div>
              </div>
              
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">Active Threads</span>
                <span className="text-white font-medium">{processorStats.threadsActive}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-white">Recent Detection Activity</h3>
          <div className="flex items-center space-x-4 text-sm">
            <div className="flex items-center">
              <Cpu className="h-4 w-4 text-cyan-400 mr-1" />
              <span className="text-gray-400">Speed:</span>
              <span className="text-white ml-1 font-medium">{processorStats.processingSpeed.toFixed(0)} FPS</span>
            </div>
            <div className="flex items-center">
              <Gauge className="h-4 w-4 text-green-400 mr-1" />
              <span className="text-gray-400">Score:</span>
              <span className={`ml-1 font-medium ${
                processorStats.performanceScore > 80 ? 'text-green-400' :
                processorStats.performanceScore > 60 ? 'text-yellow-400' : 'text-red-400'
              }`}>
                {processorStats.performanceScore.toFixed(0)}
              </span>
            </div>
          </div>
        </div>
        {recentActivity.length > 0 ? (
          <div className="space-y-4">
            {recentActivity.map((result) => (
              <div key={result.id} className="flex items-center justify-between p-4 bg-gray-700/50 rounded-lg">
                <div className="flex items-center space-x-3">
                  {result.isDeepfake ? (
                    <AlertTriangle className="h-5 w-5 text-red-400" />
                  ) : (
                    <CheckCircle className="h-5 w-5 text-green-400" />
                  )}
                  <div>
                    <p className="text-sm font-medium text-white">{result.filename}</p>
                    <p className="text-xs text-gray-400">
                      {result.timestamp.toLocaleString()} • {result.confidence.toFixed(1)}% confidence
                    </p>
                  </div>
                </div>
                <div className={`px-2 py-1 rounded text-xs font-medium ${
                  result.isDeepfake ? 'bg-red-900/50 text-red-400' : 'bg-green-900/50 text-green-400'
                }`}>
                  {result.isDeepfake ? 'THREAT' : 'SAFE'}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <Activity className="h-12 w-12 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400">No detection activity yet</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;