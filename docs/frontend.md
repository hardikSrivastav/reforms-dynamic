
# RForms Frontend Documentation

## Tech Stack

- **React 18**: For building UI components
- **Shadcn UI**: For pre-styled components and consistent design language
- **Turbopack**: For fast builds and improved developer experience
- **Tailwind CSS**: For utility-first styling
- **Next.js 14**: For project structure and routing

## Project Setup

```bash
# Create Next.js app with Turbopack
npx create-next-app@latest rforms-frontend --typescript --tailwind --use-npm --turbo

# Navigate to project
cd rforms-frontend

# Install Shadcn UI
npm install shadcn-ui @radix-ui/react-icons

# Initialize Shadcn UI
npx shadcn-ui@latest init

# Install additional dependencies
npm install axios react-hook-form zod @hookform/resolvers lucide-react react-query
```

## Project Structure

```
rforms-frontend/
├── app/
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Home page
│   ├── surveys/           # Survey pages
│   │   ├── page.tsx       # Survey list
│   │   └── [id]/          # Survey details
│   │       ├── page.tsx   # Survey form
│   │       └── results/   # Results page
│   └── api/               # Frontend API routes
├── components/
│   ├── ui/                # Shadcn UI components
│   ├── survey/            # Survey-specific components
│   ├── forms/             # Form components
│   └── layout/            # Layout components
├── lib/
│   ├── api.ts             # API client
│   ├── types.ts           # TypeScript types
│   └── utils.ts           # Utility functions
├── public/                # Static assets
└── styles/                # Global styles
```

## API Integration

Create an API client service in `lib/api.ts`:

```typescript
import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Survey endpoints
export const surveyApi = {
  getSurveys: () => api.get('/surveys'),
  getSurvey: (id: string) => api.get(`/surveys/${id}`),
};

// Session endpoints
export const sessionApi = {
  createSession: (data: { survey_id: string, user_id?: string, user_profile?: any }) => 
    api.post('/sessions', data),
  getNextQuestion: (sessionId: string) => 
    api.post(`/sessions/${sessionId}/questions/next`),
  submitResponse: (sessionId: string, questionId: string, data: any) => 
    api.post(`/sessions/${sessionId}/responses`, {
      question_id: questionId,
      response_text: data.response_text,
      response_value: data.response_value,
    }),
  endSession: (sessionId: string) => 
    api.post(`/sessions/${sessionId}/end`),
};
```

## Core Components

### 1. SurveyList Component

```tsx
// components/survey/SurveyList.tsx
import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { surveyApi } from '@/lib/api';
import Link from 'next/link';

export function SurveyList() {
  const [surveys, setSurveys] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchSurveys = async () => {
      try {
        const response = await surveyApi.getSurveys();
        setSurveys(response.data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchSurveys();
  }, []);

  if (loading) return <div className="flex justify-center p-6">Loading surveys...</div>;
  if (error) return <div className="text-red-500 p-6">Error: {error}</div>;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 p-6">
      {surveys.map((survey) => (
        <Card key={survey.id} className="flex flex-col">
          <CardHeader>
            <CardTitle>{survey.title}</CardTitle>
            <CardDescription>{survey.description}</CardDescription>
          </CardHeader>
          <CardContent className="flex-grow">
            <p className="text-sm text-gray-500">Created: {new Date(survey.created_at).toLocaleDateString()}</p>
          </CardContent>
          <CardFooter>
            <Link href={`/surveys/${survey.id}`} passHref>
              <Button className="w-full">Start Survey</Button>
            </Link>
          </CardFooter>
        </Card>
      ))}
    </div>
  );
}
```

### 2. SurveyForm Component

```tsx
// components/survey/SurveyForm.tsx
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { sessionApi } from '@/lib/api';
import QuestionRenderer from '@/components/survey/QuestionRenderer';

export function SurveyForm({ surveyId, userProfile = { age: 30 } }) {
  const router = useRouter();
  const [sessionId, setSessionId] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [complete, setComplete] = useState(false);

  // Initialize a session
  useEffect(() => {
    const initSession = async () => {
      try {
        const response = await sessionApi.createSession({
          survey_id: surveyId,
          user_profile: userProfile,
        });
        setSessionId(response.data.session_id);
        
        // Get first question
        await fetchNextQuestion(response.data.session_id);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    initSession();
  }, [surveyId]);

  const fetchNextQuestion = async (sid) => {
    try {
      setLoading(true);
      const response = await sessionApi.getNextQuestion(sid);
      
      if (response.data.session_complete) {
        setComplete(true);
        router.push(`/surveys/${surveyId}/results?sessionId=${sid}`);
        return;
      }
      
      setCurrentQuestion(response.data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitResponse = async (response) => {
    try {
      setLoading(true);
      await sessionApi.submitResponse(
        sessionId, 
        currentQuestion.id, 
        response
      );
      
      // Get next question
      await fetchNextQuestion(sessionId);
    } catch (err) {
      setError(err.message);
    }
  };

  if (loading) return <div className="flex justify-center p-6">Loading...</div>;
  if (error) return <div className="text-red-500 p-6">Error: {error}</div>;
  if (complete) return <div className="p-6">Survey complete! Redirecting to results...</div>;

  return (
    <Card className="max-w-3xl mx-auto my-8">
      <CardHeader>
        <CardTitle>{currentQuestion?.text}</CardTitle>
      </CardHeader>
      <CardContent>
        {currentQuestion && (
          <QuestionRenderer 
            question={currentQuestion} 
            onSubmit={handleSubmitResponse} 
          />
        )}
      </CardContent>
      <CardFooter className="flex justify-between">
        <div className="text-sm text-gray-500">
          Question Source: {currentQuestion?.source}
        </div>
      </CardFooter>
    </Card>
  );
}
```

### 3. QuestionRenderer Component

```tsx
// components/survey/QuestionRenderer.tsx
import { useState } from 'react';
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';
import * as z from 'zod';
import { Button } from '@/components/ui/button';
import { Form, FormControl, FormField, FormItem, FormLabel } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';

const formSchema = z.object({
  response_text: z.string(),
  response_value: z.any().optional(),
});

export default function QuestionRenderer({ question, onSubmit }) {
  const form = useForm({
    resolver: zodResolver(formSchema),
    defaultValues: {
      response_text: '',
      response_value: null,
    },
  });

  const handleSubmit = (data) => {
    onSubmit(data);
  };

  const renderQuestionByType = () => {
    switch (question.type) {
      case 'text':
        return (
          <FormField
            control={form.control}
            name="response_text"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Your answer</FormLabel>
                <FormControl>
                  <Textarea placeholder="Type your answer here..." {...field} />
                </FormControl>
              </FormItem>
            )}
          />
        );
        
      case 'number':
        return (
          <FormField
            control={form.control}
            name="response_text"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Your answer</FormLabel>
                <FormControl>
                  <Input type="number" placeholder="Enter a number..." 
                    {...field}
                    onChange={(e) => {
                      field.onChange(e);
                      form.setValue('response_value', parseFloat(e.target.value));
                    }} 
                  />
                </FormControl>
              </FormItem>
            )}
          />
        );
        
      case 'boolean':
        return (
          <FormField
            control={form.control}
            name="response_text"
            render={({ field }) => (
              <FormItem className="flex flex-row items-center justify-between rounded-lg border p-4">
                <div className="space-y-0.5">
                  <FormLabel>Yes/No</FormLabel>
                </div>
                <FormControl>
                  <Switch
                    checked={field.value === 'Yes'}
                    onCheckedChange={(checked) => {
                      const value = checked ? 'Yes' : 'No';
                      field.onChange(value);
                      form.setValue('response_value', checked);
                    }}
                  />
                </FormControl>
              </FormItem>
            )}
          />
        );
        
      case 'multiple_choice':
        return (
          <FormField
            control={form.control}
            name="response_text"
            render={({ field }) => (
              <FormItem className="space-y-3">
                <FormLabel>Select one option</FormLabel>
                <FormControl>
                  <RadioGroup
                    onValueChange={(value) => {
                      field.onChange(value);
                      form.setValue('response_value', value);
                    }}
                    defaultValue={field.value}
                    className="flex flex-col space-y-1"
                  >
                    {question.options?.map((option) => (
                      <FormItem className="flex items-center space-x-3 space-y-0" key={option.value}>
                        <FormControl>
                          <RadioGroupItem value={option.value} />
                        </FormControl>
                        <FormLabel className="font-normal">
                          {option.label}
                        </FormLabel>
                      </FormItem>
                    ))}
                  </RadioGroup>
                </FormControl>
              </FormItem>
            )}
          />
        );
        
      case 'likert':
        return (
          <FormField
            control={form.control}
            name="response_text"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Rate on a scale from 1-5</FormLabel>
                <FormControl>
                  <RadioGroup
                    onValueChange={(value) => {
                      field.onChange(value);
                      form.setValue('response_value', parseInt(value));
                    }}
                    defaultValue={field.value}
                    className="flex justify-between"
                  >
                    {[1, 2, 3, 4, 5].map((value) => (
                      <FormItem key={value} className="flex flex-col items-center">
                        <FormControl>
                          <RadioGroupItem value={value.toString()} />
                        </FormControl>
                        <FormLabel className="mt-1 text-xs">{value}</FormLabel>
                      </FormItem>
                    ))}
                  </RadioGroup>
                </FormControl>
              </FormItem>
            )}
          />
        );
        
      case 'range':
        return (
          <FormField
            control={form.control}
            name="response_value"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Select a value</FormLabel>
                <FormControl>
                  <div className="space-y-4">
                    <Slider 
                      min={question.min_value || 0}
                      max={question.max_value || 100}
                      step={1}
                      defaultValue={[50]}
                      onValueChange={(value) => {
                        form.setValue('response_value', value[0]);
                        form.setValue('response_text', value[0].toString());
                      }}
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>{question.min_value || 0}</span>
                      <span>{question.max_value || 100}</span>
                    </div>
                  </div>
                </FormControl>
              </FormItem>
            )}
          />
        );
        
      default:
        return (
          <FormField
            control={form.control}
            name="response_text"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Your answer</FormLabel>
                <FormControl>
                  <Textarea placeholder="Type your answer here..." {...field} />
                </FormControl>
              </FormItem>
            )}
          />
        );
    }
  };

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-8">
        {renderQuestionByType()}
        <Button type="submit" className="w-full">
          Next Question
        </Button>
      </form>
    </Form>
  );
}
```

## Pages

### Home Page (app/page.tsx)

```tsx
// app/page.tsx
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import Link from 'next/link';

export default function Home() {
  return (
    <main className="container mx-auto px-4 py-12">
      <div className="flex flex-col items-center justify-center min-h-[70vh] text-center">
        <h1 className="text-4xl font-bold tracking-tight mb-6">RForms Dynamic Survey Engine</h1>
        <p className="text-xl text-gray-600 max-w-2xl mb-8">
          An adaptive question engine that minimizes latency and maximizes data quality 
          while asking the fewest possible questions.
        </p>
        
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle>Get Started</CardTitle>
            <CardDescription>
              Take a survey or view available surveys
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-500 mb-4">
              Our adaptive surveys use AI to personalize questions based on your responses,
              making the experience more efficient and relevant.
            </p>
          </CardContent>
          <CardFooter>
            <Link href="/surveys" className="w-full">
              <Button className="w-full">Browse Surveys</Button>
            </Link>
          </CardFooter>
        </Card>
      </div>
    </main>
  );
}
```

### Surveys List Page (app/surveys/page.tsx)

```tsx
// app/surveys/page.tsx
import { SurveyList } from '@/components/survey/SurveyList';

export default function SurveysPage() {
  return (
    <div className="container mx-auto px-4 py-12">
      <h1 className="text-3xl font-bold mb-8">Available Surveys</h1>
      <SurveyList />
    </div>
  );
}
```

### Survey Form Page (app/surveys/[id]/page.tsx)

```tsx
// app/surveys/[id]/page.tsx
import { SurveyForm } from '@/components/survey/SurveyForm';

export default function SurveyPage({ params }) {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">Survey</h1>
      <SurveyForm surveyId={params.id} />
    </div>
  );
}
```

### Results Page (app/surveys/[id]/results/page.tsx)

```tsx
// app/surveys/[id]/results/page.tsx
'use client';

import { useEffect, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { sessionApi } from '@/lib/api';

export default function ResultsPage({ params }) {
  const searchParams = useSearchParams();
  const sessionId = searchParams.get('sessionId');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchResults = async () => {
      if (!sessionId) return;
      
      try {
        const response = await sessionApi.endSession(sessionId);
        setResults(response.data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, [sessionId]);

  if (loading) return <div className="flex justify-center p-6">Loading results...</div>;
  if (error) return <div className="text-red-500 p-6">Error: {error}</div>;
  if (!results) return <div className="p-6">No results found.</div>;

  return (
    <div className="container mx-auto px-4 py-12">
      <h1 className="text-3xl font-bold mb-8">Survey Results</h1>
      
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Survey Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4">
            <div className="flex justify-between">
              <span>Total Questions:</span>
              <span>{results.total_questions}</span>
            </div>
            <div className="flex justify-between">
              <span>Duration:</span>
              <span>{results.duration_seconds ? Math.round(results.duration_seconds) + 's' : 'N/A'}</span>
            </div>
          </div>
        </CardContent>
      </Card>
      
      <h2 className="text-2xl font-bold mt-8 mb-4">Metrics</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {Object.entries(results.metrics || {}).map(([key, value]) => (
          <Card key={key}>
            <CardHeader>
              <CardTitle>{key.replace('_', ' ').toUpperCase()}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Score:</span>
                  <span>{(value.score * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Confidence:</span>
                  <span>{(value.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Questions:</span>
                  <span>{value.questions_asked}</span>
                </div>
                
                <div className="w-full bg-gray-200 rounded-full h-2.5 mt-4">
                  <div 
                    className="bg-primary h-2.5 rounded-full" 
                    style={{ width: `${value.score * 100}%` }}
                  ></div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
```

## Theme Configuration

Set up a custom theme with Shadcn UI by modifying your `tailwind.config.js`:

```javascript
// tailwind.config.js
const { fontFamily } = require("tailwindcss/defaultTheme")

/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      fontFamily: {
        sans: ["var(--font-sans)", ...fontFamily.sans],
      },
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: 0 },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: 0 },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}
```

## Deployment

For deployment, you can set up a Next.js application on Vercel:

1. Create a Git repository and push your code
2. Connect your repository to Vercel
3. Configure environment variables:
   - `NEXT_PUBLIC_API_BASE_URL`: URL of your API (e.g., https://api.example.com/api)

## Development Notes

1. Ensure your API is running and accessible from the frontend (CORS configuration)
2. For local development, use environment variables in `.env.local`:
   ```
   NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api
   ```
3. Use Turbopack during development (`next dev --turbo`) for faster reloads
4. Consider implementing authentication for protected surveys


