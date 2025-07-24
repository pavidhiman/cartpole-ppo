import axios from 'axios';
import { DEBUG_BYPASS_AUTH } from './config.debug';
import config from './config.json';
import {
  getAccessToken,
  handleTokenRefresh,
  tokenExpired,
} from './utils/Library';


const api = axios.create({
  baseURL: config.ROOT_URL,
});

api.interceptors.request.use(
  async (config) => {
    let token = await getAccessToken();
    if (tokenExpired(token)) {
      token = await handleTokenRefresh();
    }
    // Always set the Authorization header if a token exists
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  },
);

// debugging to find all 401 errors
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401 && DEBUG_BYPASS_AUTH) {
      const { method, url } = err.config;
      console.log(`[401-mock] ${method.toUpperCase()} ${url}`);

      // very small stubs – shape them per endpoint you hit
      if (url.includes('/Patient/ValidateUser')) {
        return Promise.resolve({ status: 200, data: { userID: 1, patientID: 99 } });
      }
      if (url.includes('/Patient/SaveEULA')) {
        return Promise.resolve({ status: 200, data: {} });
      }
    }
    return Promise.reject(err);
  }
);


export default api;


