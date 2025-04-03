import React from "react";

const Button = ({ children, onClick, className, variant = "primary" }) => {
  const baseStyles = "px-4 py-2 rounded-lg text-sm font-medium transition";
  const variants = {
    primary: "bg-blue-600 text-white hover:bg-blue-700",
    secondary: "bg-gray-200 text-gray-800 hover:bg-gray-300",
    danger: "bg-red-600 text-white hover:bg-red-700",
  };

  return (
    <button className={`${baseStyles} ${variants[variant]} ${className}`} onClick={onClick}>
      {children}
    </button>
  );
};

export default Button;
