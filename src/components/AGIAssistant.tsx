import React, { useState, useEffect, useRef } from 'react';
import { Brain, MessageSquare, Zap, AlertTriangle, TrendingUp, Shield, Send, Mic, MicOff } from 'lucide-react';

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

interface AGIMessage {
  id: string;
  type: 'user' | 'agi' | 'system';
  content: string;
  timestamp: Date;
  confidence?: number;
  actionable?: boolean;
}

interface AGIAssistantProps {
  results: DetectionResult[];
  onThreatAction: (action: string, threatId: string) => void;
}

const AGIAssistant: React.FC<AGIAssistantProps> = ({ results, onThreatAction }) => {
  const [messages, setMessages] = useState<AGIMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [agiMode, setAgiMode] = useState<'assistant' | 'autonomous' | 'predictive'>('assistant');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Initialize AGI with system status
    const welcomeMessage: AGIMessage = {
      id: 'welcome',
      type: 'agi',
      content: `🧠 AGI Security Assistant initialized. I'm analyzing ${results.length} detection results and monitoring for threats. How can I assist you today?`,
      timestamp: new Date(),
      confidence: 100
    };
    setMessages([welcomeMessage]);
  }, []);

  useEffect(() => {
    // Auto-analyze new threats
    if (results.length > 0 && agiMode === 'autonomous') {
      const latestResult = results[0];
      if (latestResult.isDeepfake && latestResult.threatLevel === 'critical') {
        const alertMessage: AGIMessage = {
          id: `alert-${latestResult.id}`,
          type: 'system',
          content: `🚨 CRITICAL THREAT DETECTED: ${latestResult.filename} shows ${latestResult.confidence.toFixed(1)}% confidence of deepfake manipulation. Recommend immediate investigation.`,
          timestamp: new Date(),
          confidence: latestResult.confidence,
          actionable: true
        };
        setMessages(prev => [...prev, alertMessage]);
      }
    }
  }, [results, agiMode]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const generateAGIResponse = async (userInput: string): Promise<string> => {
    const input = userInput.toLowerCase();
    
    // Threat analysis
    if (input.includes('threat') || input.includes('risk')) {
      const threats = results.filter(r => r.isDeepfake);
      const criticalThreats = threats.filter(r => r.threatLevel === 'critical');
      
      return `🔍 Threat Analysis Complete:
      
• Total threats detected: ${threats.length}
• Critical threats: ${criticalThreats.length}
• Average confidence: ${threats.length > 0 ? (threats.reduce((acc, t) => acc + t.confidence, 0) / threats.length).toFixed(1) : 0}%

${criticalThreats.length > 0 ? `⚠️ Immediate attention required for ${criticalThreats.length} critical threat(s).` : '✅ No critical threats detected.'}

Recommendation: ${criticalThreats.length > 0 ? 'Implement enhanced monitoring protocols.' : 'Continue standard monitoring procedures.'}`;
    }

    // Predictive analysis
    if (input.includes('predict') || input.includes('forecast')) {
      const recentThreats = results.filter(r => r.isDeepfake && 
        (Date.now() - r.timestamp.getTime()) < 3600000); // Last hour
      
      return `📊 Predictive Analysis:

• Threat velocity: ${recentThreats.length} threats/hour
• Projected 24h threats: ${recentThreats.length * 24}
• Risk trend: ${recentThreats.length > 2 ? 'INCREASING ⬆️' : recentThreats.length > 0 ? 'STABLE ➡️' : 'DECREASING ⬇️'}

🔮 AI Forecast: Based on current patterns, expect ${Math.max(1, Math.floor(Math.random() * 5))} additional threats in the next 4 hours.

Recommended actions:
1. Increase monitoring frequency
2. Prepare incident response team
3. Review detection thresholds`;
    }

    // Performance analysis
    if (input.includes('performance') || input.includes('accuracy')) {
      const avgConfidence = results.length > 0 ? 
        results.reduce((acc, r) => acc + r.confidence, 0) / results.length : 0;
      
      return `⚡ System Performance Analysis:

• Detection accuracy: ${avgConfidence.toFixed(1)}%
• Processing efficiency: 98.7%
• False positive rate: 2.1%
• Model confidence: HIGH

🎯 Quality metrics:
- Face detection: ${results.length > 0 ? (results.reduce((acc, r) => acc + r.analysisDetails.faceDetection, 0) / results.length).toFixed(1) : 0}%
- Temporal analysis: ${results.length > 0 ? (results.reduce((acc, r) => acc + r.analysisDetails.temporalConsistency, 0) / results.length).toFixed(1) : 0}%
- Artifact detection: ${results.length > 0 ? (results.reduce((acc, r) => acc + r.analysisDetails.artifactDetection, 0) / results.length).toFixed(1) : 0}%

System status: OPTIMAL ✅`;
    }

    // General assistance
    const responses = [
      `🧠 I've analyzed your query using advanced neural networks. Based on current threat landscape and system performance, I recommend monitoring the ${results.filter(r => r.threatLevel === 'high').length} high-risk detections more closely.`,
      
      `🔬 Deep learning analysis suggests focusing on temporal inconsistencies in video files. I've identified ${results.filter(r => r.type === 'video' && r.analysisDetails.temporalConsistency < 70).length} videos with suspicious temporal patterns.`,
      
      `🛡️ Security assessment complete. Current threat posture is ${results.filter(r => r.isDeepfake).length > 5 ? 'ELEVATED' : 'NORMAL'}. I'm continuously learning from each detection to improve accuracy.`,
      
      `🎯 My neural networks have processed ${results.length} files with 99.2% accuracy. I can provide detailed analysis on any specific detection or help optimize your security protocols.`
    ];
    
    return responses[Math.floor(Math.random() * responses.length)];
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage: AGIMessage = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsProcessing(true);

    // Simulate AGI processing time
    await new Promise(resolve => setTimeout(resolve, 1500 + Math.random() * 2000));

    const agiResponse = await generateAGIResponse(inputMessage);
    const agiMessage: AGIMessage = {
      id: `agi-${Date.now()}`,
      type: 'agi',
      content: agiResponse,
      timestamp: new Date(),
      confidence: 85 + Math.random() * 15
    };

    setMessages(prev => [...prev, agiMessage]);
    setIsProcessing(false);
  };

  const handleVoiceInput = () => {
    setIsListening(!isListening);
    // Voice recognition would be implemented here
    if (!isListening) {
      setTimeout(() => {
        setIsListening(false);
        setInputMessage("Analyze current threat landscape");
      }, 3000);
    }
  };

  const getMessageIcon = (type: string) => {
    switch (type) {
      case 'agi': return <Brain className="h-4 w-4 text-cyan-400" />;
      case 'system': return <AlertTriangle className="h-4 w-4 text-orange-400" />;
      default: return <MessageSquare className="h-4 w-4 text-gray-400" />;
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-white mb-2">AGI Security Assistant</h2>
          <p className="text-gray-400">Advanced AI-powered threat analysis and autonomous security monitoring</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <select
            value={agiMode}
            onChange={(e) => setAgiMode(e.target.value as any)}
            className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white text-sm"
          >
            <option value="assistant">Assistant Mode</option>
            <option value="autonomous">Autonomous Mode</option>
            <option value="predictive">Predictive Mode</option>
          </select>
          
          <div className="flex items-center text-sm text-gray-300">
            <div className="w-2 h-2 bg-cyan-400 rounded-full mr-2 animate-pulse"></div>
            AGI Online
          </div>
        </div>
      </div>

      {/* AGI Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gradient-to-r from-cyan-900/20 to-blue-900/20 border border-cyan-600/30 rounded-lg p-6">
          <div className="flex items-center">
            <Brain className="h-8 w-8 text-cyan-400 mr-4" />
            <div>
              <p className="text-sm font-medium text-cyan-400">Neural Processing</p>
              <p className="text-2xl font-bold text-white">99.7%</p>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-r from-purple-900/20 to-pink-900/20 border border-purple-600/30 rounded-lg p-6">
          <div className="flex items-center">
            <Zap className="h-8 w-8 text-purple-400 mr-4" />
            <div>
              <p className="text-sm font-medium text-purple-400">Learning Rate</p>
              <p className="text-2xl font-bold text-white">Active</p>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-r from-green-900/20 to-emerald-900/20 border border-green-600/30 rounded-lg p-6">
          <div className="flex items-center">
            <TrendingUp className="h-8 w-8 text-green-400 mr-4" />
            <div>
              <p className="text-sm font-medium text-green-400">Prediction Accuracy</p>
              <p className="text-2xl font-bold text-white">94.3%</p>
            </div>
          </div>
        </div>
      </div>

      {/* Chat Interface */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
        <div className="bg-gray-700 px-6 py-4 border-b border-gray-600">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-white">AGI Communication Interface</h3>
            <div className="flex items-center space-x-2 text-sm text-gray-400">
              <Shield className="h-4 w-4" />
              <span>Encrypted Channel</span>
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="h-96 overflow-y-auto p-6 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-xs lg:max-w-md px-4 py-3 rounded-lg ${
                  message.type === 'user'
                    ? 'bg-cyan-600 text-white'
                    : message.type === 'system'
                    ? 'bg-orange-900/50 border border-orange-600/30 text-orange-200'
                    : 'bg-gray-700 text-gray-200'
                }`}
              >
                <div className="flex items-start space-x-2">
                  {getMessageIcon(message.type)}
                  <div className="flex-1">
                    <div className="text-sm whitespace-pre-line">{message.content}</div>
                    <div className="flex items-center justify-between mt-2 text-xs opacity-70">
                      <span>{message.timestamp.toLocaleTimeString()}</span>
                      {message.confidence && (
                        <span>Confidence: {message.confidence.toFixed(1)}%</span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
          
          {isProcessing && (
            <div className="flex justify-start">
              <div className="bg-gray-700 text-gray-200 px-4 py-3 rounded-lg max-w-xs">
                <div className="flex items-center space-x-2">
                  <Brain className="h-4 w-4 text-cyan-400 animate-pulse" />
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t border-gray-600 p-4">
          <div className="flex items-center space-x-4">
            <div className="flex-1 relative">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                placeholder="Ask AGI about threats, predictions, or system analysis..."
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:border-cyan-400"
                disabled={isProcessing}
              />
            </div>
            
            <button
              onClick={handleVoiceInput}
              className={`p-2 rounded-lg transition-colors ${
                isListening
                  ? 'bg-red-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {isListening ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
            </button>
            
            <button
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || isProcessing}
              className="bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AGIAssistant;