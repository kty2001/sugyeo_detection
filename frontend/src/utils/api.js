import axios from 'axios';

// const API_URL = process.env.REACT_APP_API_URL || '';
const API_URL = 'http://localhost:8000';

export const processSoyanalysis = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/api/soyanalysis/process`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
    });
    console.log('soyanalysis 응답:', response.data)
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || '두유 분석 중 오류가 발생했습니다.');
  }
}; 

export const processSoymilk = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/api/soymilk/process`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
    });
    console.log('soymilk 응답:', response.data)
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || '두유 자르기 중 오류가 발생했습니다.');
  }
}; 
