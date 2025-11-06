import React from 'react';
import { Link } from 'react-router-dom';
import { FaWaveSquare } from 'react-icons/fa';

const HomePage = () => {
  return (
    <div className="p-lg">
      <h1 className="text-textPrimary text-3xl mb-lg">수계 탐지 시스템</h1>
      <p className="text-textSecondary mb-xl max-w-[800px] leading-relaxed">
        AI 기반 수계 탐지 및 인간 탐지 시스템입니다.<br />
        수계와 인간의 위치를 실시간으로 탐지하여 위험한 상황을 예방합니다.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-lg mt-xl">
        <Link 
          to="/sugyeo"
          className="bg-surface rounded-md p-lg shadow-md hover:-translate-y-1 hover:shadow-lg transition-all duration-fast flex flex-col items-center text-center"
        >
          <div className="text-5xl text-primary mb-md">
            <FaWaveSquare />
          </div>
          <h3 className="text-textPrimary text-xl mb-md">수계 탐지</h3>
          <p className="text-textSecondary text-sm">
            수계와 사람을 탐지합니다.
          </p>
        </Link>
      </div>
      
    </div>
  );
};

export default HomePage; 