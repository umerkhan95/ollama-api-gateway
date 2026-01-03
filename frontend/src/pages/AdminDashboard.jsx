import React, { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import StatsCard from '../components/StatsCard';
import APIKeyForm from '../components/APIKeyForm';
import { BarChart, DoughnutChart } from '../components/Charts';
import { Key, Users, Activity, Trash2, Plus, RefreshCw, Clock, TrendingUp, BarChart2, Eye, EyeOff, Copy, Check } from 'lucide-react';
import { format } from 'date-fns';

const AdminDashboard = () => {
  const [apiKeys, setApiKeys] = useState([]);
  const [stats, setStats] = useState(null);
  const [userStats, setUserStats] = useState(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview'); // 'overview', 'users', 'keys'
  const [revealedKeys, setRevealedKeys] = useState({}); // Store revealed keys by ID
  const [copiedKeyId, setCopiedKeyId] = useState(null); // Track which key was copied

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    const [keysResult, statsResult, userStatsResult] = await Promise.all([
      apiService.getApiKeys(),
      apiService.getDetailedStats(),
      apiService.getUserStats()
    ]);

    if (keysResult.success) {
      setApiKeys(keysResult.data);
    }
    if (statsResult.success) {
      setStats(statsResult.data);
    }
    if (userStatsResult.success) {
      setUserStats(userStatsResult.data);
    }
    setLoading(false);
  };

  const handleRevokeKey = async (keyId, keyName) => {
    if (!confirm(`Are you sure you want to revoke the API key "${keyName}"?`)) {
      return;
    }

    const result = await apiService.revokeApiKey(keyId);
    if (result.success) {
      fetchData();
    } else {
      alert(result.error);
    }
  };

  const handleRevealKey = async (keyId) => {
    // If already revealed, hide it
    if (revealedKeys[keyId]) {
      setRevealedKeys(prev => {
        const newKeys = { ...prev };
        delete newKeys[keyId];
        return newKeys;
      });
      return;
    }

    // Fetch the full key
    const result = await apiService.revealApiKey(keyId);
    if (result.success) {
      setRevealedKeys(prev => ({
        ...prev,
        [keyId]: result.data.key
      }));
    } else {
      alert(result.error);
    }
  };

  const handleCopyKey = async (keyId, key) => {
    try {
      await navigator.clipboard.writeText(key);
      setCopiedKeyId(keyId);
      setTimeout(() => setCopiedKeyId(null), 2000);
    } catch (err) {
      alert('Failed to copy to clipboard');
    }
  };

  // Chart data for user requests (bar chart)
  const getUserRequestsChartData = () => {
    if (!userStats?.users) return null;
    
    const sortedUsers = [...userStats.users]
      .sort((a, b) => b.requests_24h - a.requests_24h)
      .slice(0, 10); // Top 10 users

    return {
      labels: sortedUsers.map(u => u.name),
      datasets: [
        {
          label: 'Requests (24h)',
          data: sortedUsers.map(u => u.requests_24h),
          backgroundColor: 'rgba(59, 130, 246, 0.8)',
          borderColor: 'rgba(59, 130, 246, 1)',
          borderWidth: 1,
        },
        {
          label: 'Requests (7d)',
          data: sortedUsers.map(u => u.requests_7d),
          backgroundColor: 'rgba(139, 92, 246, 0.8)',
          borderColor: 'rgba(139, 92, 246, 1)',
          borderWidth: 1,
        }
      ],
    };
  };

  // Chart data for average response time per user
  const getResponseTimeChartData = () => {
    if (!userStats?.users) return null;
    
    const sortedUsers = [...userStats.users]
      .filter(u => u.avg_response_time > 0)
      .sort((a, b) => b.avg_response_time - a.avg_response_time)
      .slice(0, 10);

    return {
      labels: sortedUsers.map(u => u.name),
      datasets: [
        {
          label: 'Avg Response Time (s)',
          data: sortedUsers.map(u => u.avg_response_time),
          backgroundColor: sortedUsers.map((_, i) => {
            const colors = [
              'rgba(239, 68, 68, 0.8)',
              'rgba(249, 115, 22, 0.8)',
              'rgba(234, 179, 8, 0.8)',
              'rgba(34, 197, 94, 0.8)',
              'rgba(59, 130, 246, 0.8)',
              'rgba(139, 92, 246, 0.8)',
              'rgba(236, 72, 153, 0.8)',
              'rgba(20, 184, 166, 0.8)',
              'rgba(168, 85, 247, 0.8)',
              'rgba(251, 146, 60, 0.8)',
            ];
            return colors[i % colors.length];
          }),
          borderWidth: 1,
        }
      ],
    };
  };

  // Chart data for rate limit usage
  const getRateLimitChartData = () => {
    if (!userStats?.users) return null;
    
    const usersWithUsage = userStats.users.filter(u => u.rate_limit_usage > 0);
    
    return {
      labels: usersWithUsage.map(u => u.name),
      datasets: [
        {
          label: 'Rate Limit Usage (%)',
          data: usersWithUsage.map(u => u.rate_limit_usage),
          backgroundColor: usersWithUsage.map(u => {
            if (u.rate_limit_usage >= 90) return 'rgba(239, 68, 68, 0.8)';
            if (u.rate_limit_usage >= 70) return 'rgba(249, 115, 22, 0.8)';
            if (u.rate_limit_usage >= 50) return 'rgba(234, 179, 8, 0.8)';
            return 'rgba(34, 197, 94, 0.8)';
          }),
          borderWidth: 1,
        }
      ],
    };
  };

  // Role distribution chart
  const getRoleDistributionData = () => {
    if (!userStats?.users) return null;
    
    const adminCount = userStats.users.filter(u => u.role === 'admin').length;
    const userCount = userStats.users.filter(u => u.role === 'user').length;

    return {
      labels: ['Admin', 'User'],
      datasets: [
        {
          data: [adminCount, userCount],
          backgroundColor: [
            'rgba(139, 92, 246, 0.8)',
            'rgba(34, 197, 94, 0.8)',
          ],
          borderColor: [
            'rgba(139, 92, 246, 1)',
            'rgba(34, 197, 94, 1)',
          ],
          borderWidth: 2,
        }
      ],
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#9CA3AF',
        }
      },
    },
    scales: {
      x: {
        ticks: { color: '#9CA3AF' },
        grid: { color: 'rgba(156, 163, 175, 0.1)' }
      },
      y: {
        ticks: { color: '#9CA3AF' },
        grid: { color: 'rgba(156, 163, 175, 0.1)' }
      }
    }
  };

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          color: '#9CA3AF',
        }
      },
    },
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-gray-600 dark:text-gray-400">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Admin Dashboard</h1>
            <p className="text-gray-600 dark:text-gray-400 mt-2">Manage API keys and monitor usage</p>
          </div>
          <div className="flex space-x-4">
            <button
              onClick={fetchData}
              className="flex items-center space-x-2 px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600"
            >
              <RefreshCw className="h-5 w-5" />
              <span>Refresh</span>
            </button>
            <button
              onClick={() => setShowCreateForm(true)}
              className="flex items-center space-x-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
            >
              <Plus className="h-5 w-5" />
              <span>Create API Key</span>
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="mb-6 border-b border-gray-200 dark:border-gray-700">
          <nav className="-mb-px flex space-x-8">
            {[
              { id: 'overview', label: 'Overview', icon: Activity },
              { id: 'users', label: 'User Analytics', icon: Users },
              { id: 'keys', label: 'API Keys', icon: Key },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                }`}
              >
                <tab.icon className="h-5 w-5" />
                <span>{tab.label}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <>
            {/* Summary Stats */}
            {userStats?.summary && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
                <StatsCard
                  title="Total Users"
                  value={userStats.summary.total_users}
                  icon={Users}
                  description="Active API keys"
                />
                <StatsCard
                  title="Total Requests"
                  value={userStats.summary.total_requests}
                  icon={Activity}
                  description="All time"
                />
                <StatsCard
                  title="Requests (24h)"
                  value={userStats.summary.total_requests_24h}
                  icon={TrendingUp}
                  description="Last 24 hours"
                />
                <StatsCard
                  title="Requests (7d)"
                  value={userStats.summary.total_requests_7d}
                  icon={BarChart2}
                  description="Last 7 days"
                />
                <StatsCard
                  title="Avg Response Time"
                  value={`${userStats.summary.avg_response_time}s`}
                  icon={Clock}
                  description="Across all users"
                />
              </div>
            )}

            {/* Charts Row 1 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              {/* User Requests Chart */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Top Users by Requests
                </h3>
                <div className="h-80">
                  {getUserRequestsChartData() && (
                    <BarChart data={getUserRequestsChartData()} options={chartOptions} />
                  )}
                </div>
              </div>

              {/* Response Time Chart */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Average Response Time by User
                </h3>
                <div className="h-80">
                  {getResponseTimeChartData() && (
                    <BarChart data={getResponseTimeChartData()} options={chartOptions} />
                  )}
                </div>
              </div>
            </div>

            {/* Charts Row 2 */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
              {/* Rate Limit Usage */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 lg:col-span-2">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Rate Limit Usage (This Hour)
                </h3>
                <div className="h-64">
                  {getRateLimitChartData() && getRateLimitChartData().labels.length > 0 ? (
                    <BarChart data={getRateLimitChartData()} options={chartOptions} />
                  ) : (
                    <div className="flex items-center justify-center h-full text-gray-500 dark:text-gray-400">
                      No rate limit usage data available
                    </div>
                  )}
                </div>
              </div>

              {/* Role Distribution */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  User Role Distribution
                </h3>
                <div className="h-64">
                  {getRoleDistributionData() && (
                    <DoughnutChart data={getRoleDistributionData()} options={doughnutOptions} />
                  )}
                </div>
              </div>
            </div>
          </>
        )}

        {/* User Analytics Tab */}
        {activeTab === 'users' && userStats?.users && (
          <div className="space-y-6">
            {/* User Stats Table */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden">
              <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">User Statistics</h2>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead className="bg-gray-50 dark:bg-gray-700">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        User
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        Role
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        Total Requests
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        24h
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        7d
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        Avg Response
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        Rate Limit
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        Last Active
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                    {userStats.users.map((user) => (
                      <tr key={user.id} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900 dark:text-white">
                            {user.name}
                          </div>
                          <div className="text-xs text-gray-500 dark:text-gray-400 font-mono">
                            {user.key_preview}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                            user.role === 'admin'
                              ? 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200'
                              : 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                          }`}>
                            {user.role}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white font-semibold">
                          {user.total_requests.toLocaleString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                          {user.requests_24h.toLocaleString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                          {user.requests_7d.toLocaleString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                          {user.avg_response_time}s
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className="w-16 bg-gray-200 dark:bg-gray-600 rounded-full h-2 mr-2">
                              <div
                                className={`h-2 rounded-full ${
                                  user.rate_limit_usage >= 90
                                    ? 'bg-red-500'
                                    : user.rate_limit_usage >= 70
                                    ? 'bg-orange-500'
                                    : user.rate_limit_usage >= 50
                                    ? 'bg-yellow-500'
                                    : 'bg-green-500'
                                }`}
                                style={{ width: `${Math.min(user.rate_limit_usage, 100)}%` }}
                              />
                            </div>
                            <span className="text-xs text-gray-500 dark:text-gray-400">
                              {user.requests_this_hour}/{user.rate_limit}
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                          {user.last_request
                            ? format(new Date(user.last_request), 'MMM d, HH:mm')
                            : 'Never'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Individual User Cards with Model Usage */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {userStats.users.slice(0, 6).map((user) => (
                <div key={user.id} className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{user.name}</h3>
                      <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
                        user.role === 'admin'
                          ? 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200'
                          : 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                      }`}>
                        {user.role}
                      </span>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-primary-600 dark:text-primary-400">
                        {user.total_requests}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">Total Requests</div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500 dark:text-gray-400">Avg Response Time</span>
                      <span className="font-medium text-gray-900 dark:text-white">{user.avg_response_time}s</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500 dark:text-gray-400">Rate Limit Usage</span>
                      <span className="font-medium text-gray-900 dark:text-white">{user.rate_limit_usage}%</span>
                    </div>

                    {/* Model Usage */}
                    {Object.keys(user.requests_by_model).length > 0 && (
                      <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                        <h4 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase mb-2">
                          Models Used
                        </h4>
                        <div className="space-y-1">
                          {Object.entries(user.requests_by_model)
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 3)
                            .map(([model, count]) => (
                              <div key={model} className="flex justify-between text-sm">
                                <span className="text-gray-600 dark:text-gray-300 truncate max-w-32">{model}</span>
                                <span className="font-medium text-gray-900 dark:text-white">{count}</span>
                              </div>
                            ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* API Keys Tab */}
        {activeTab === 'keys' && (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">API Keys</h2>
              <button
                onClick={fetchData}
                className="flex items-center space-x-1 text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400"
              >
                <RefreshCw className="h-4 w-4" />
                <span className="text-sm">Refresh</span>
              </button>
            </div>

            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead className="bg-gray-50 dark:bg-gray-700">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      Name
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      Role
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      Rate Limit
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      Created
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      Last Used
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                  {apiKeys.map((key) => (
                    <React.Fragment key={key.id}>
                    <tr className="hover:bg-gray-50 dark:hover:bg-gray-700">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-gray-900 dark:text-white">
                          {key.name}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400 font-mono">
                          {key.key_preview || 'N/A'}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                          key.role === 'admin'
                            ? 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200'
                            : 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                        }`}>
                          {key.role}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                        {key.rate_limit} req/hr
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {format(new Date(key.created_at), 'MMM d, yyyy')}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {key.last_used_at ? format(new Date(key.last_used_at), 'MMM d, HH:mm') : 'Never'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm">
                        <div className="flex items-center justify-end space-x-2">
                          <button
                            onClick={() => handleRevealKey(key.id)}
                            className="text-gray-600 hover:text-primary-600 dark:text-gray-400 dark:hover:text-primary-400"
                            title={revealedKeys[key.id] ? "Hide API Key" : "Show API Key"}
                          >
                            {revealedKeys[key.id] ? (
                              <EyeOff className="h-5 w-5" />
                            ) : (
                              <Eye className="h-5 w-5" />
                            )}
                          </button>
                          {revealedKeys[key.id] && (
                            <button
                              onClick={() => handleCopyKey(key.id, revealedKeys[key.id])}
                              className={`${
                                copiedKeyId === key.id
                                  ? 'text-green-600 dark:text-green-400'
                                  : 'text-gray-600 hover:text-primary-600 dark:text-gray-400 dark:hover:text-primary-400'
                              }`}
                              title={copiedKeyId === key.id ? "Copied!" : "Copy API Key"}
                            >
                              {copiedKeyId === key.id ? (
                                <Check className="h-5 w-5" />
                              ) : (
                                <Copy className="h-5 w-5" />
                              )}
                            </button>
                          )}
                          <button
                            onClick={() => handleRevokeKey(key.id, key.name)}
                            className="text-red-600 hover:text-red-900 dark:text-red-400 dark:hover:text-red-300"
                            title="Revoke API Key"
                          >
                            <Trash2 className="h-5 w-5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                    {/* Revealed Key Row */}
                    {revealedKeys[key.id] && (
                      <tr className="bg-gray-50 dark:bg-gray-700">
                        <td colSpan="6" className="px-6 py-3">
                          <div className="flex items-center space-x-3">
                            <span className="text-xs font-medium text-gray-500 dark:text-gray-400">API Key:</span>
                            <code className="flex-1 px-3 py-2 bg-gray-100 dark:bg-gray-800 rounded text-sm font-mono text-gray-800 dark:text-gray-200 break-all">
                              {revealedKeys[key.id]}
                            </code>
                            <button
                              onClick={() => handleCopyKey(key.id, revealedKeys[key.id])}
                              className={`flex items-center space-x-1 px-3 py-1 rounded text-sm ${
                                copiedKeyId === key.id
                                  ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                                  : 'bg-primary-100 text-primary-700 dark:bg-primary-900 dark:text-primary-300 hover:bg-primary-200 dark:hover:bg-primary-800'
                              }`}
                            >
                              {copiedKeyId === key.id ? (
                                <>
                                  <Check className="h-4 w-4" />
                                  <span>Copied!</span>
                                </>
                              ) : (
                                <>
                                  <Copy className="h-4 w-4" />
                                  <span>Copy</span>
                                </>
                              )}
                            </button>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
                </tbody>
              </table>
            </div>

            {apiKeys.length === 0 && (
              <div className="text-center py-12">
                <Key className="mx-auto h-12 w-12 text-gray-400" />
                <p className="mt-2 text-gray-600 dark:text-gray-400">No API keys yet</p>
                <button
                  onClick={() => setShowCreateForm(true)}
                  className="mt-4 text-primary-600 hover:text-primary-700 dark:text-primary-400"
                >
                  Create your first API key
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {showCreateForm && (
        <APIKeyForm
          onClose={() => setShowCreateForm(false)}
          onSuccess={fetchData}
        />
      )}
    </div>
  );
};

export default AdminDashboard;
