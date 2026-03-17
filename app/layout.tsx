import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'March MathNess — ML Bracket Predictor',
  description: 'NCAA bracket predictions powered by gradient descent, logistic regression, and Monte Carlo simulation',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#020817] text-white antialiased">
        {children}
      </body>
    </html>
  );
}
