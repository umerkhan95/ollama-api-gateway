import React, { createContext, useContext, useState, useEffect } from 'react';
import { apiService } from '../services/api';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    const apiKey = localStorage.getItem('apiKey');
    if (apiKey) {
      const result = await apiService.verifyApiKey(apiKey);
      if (result.success) {
        // Extract user info from the response
        // The API returns all keys, find the one matching our key
        const keysList = result.data.api_keys || [];
        const currentKey = keysList.find(k => k.key_preview.startsWith(apiKey.substring(0, 20)));
        
        setUser({
          apiKey: apiKey,
          name: currentKey?.name || 'User',
          role: currentKey?.role || 'user',
          rateLimit: currentKey?.rate_limit || 100
        });
      } else {
        localStorage.removeItem('apiKey');
      }
    }
    setLoading(false);
  };

  const login = async (apiKey) => {
    const result = await apiService.verifyApiKey(apiKey);
    if (result.success) {
      localStorage.setItem('apiKey', apiKey);
      
      // Extract user info from the response
      const keysList = result.data.api_keys || [];
      const currentKey = keysList.find(k => k.key_preview.startsWith(apiKey.substring(0, 20)));
      
      setUser({
        apiKey: apiKey,
        name: currentKey?.name || 'User',
        role: currentKey?.role || 'user',
        rateLimit: currentKey?.rate_limit || 100
      });
      return { success: true };
    }
    return result;
  };

  const logout = () => {
    localStorage.removeItem('apiKey');
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, logout, checkAuth }}>
      {children}
    </AuthContext.Provider>
  );
};
