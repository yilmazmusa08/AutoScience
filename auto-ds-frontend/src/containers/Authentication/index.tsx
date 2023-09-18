import React from "react";
import { Tabs, Form, Input, Button } from "antd";
import { useApp } from "../../context/app.context";
import "./index.css";

const Authentication: React.FC = () => {
  const [loginForm] = Form.useForm();
  const [registerForm] = Form.useForm();
  const { login, register } = useApp();

  const handleLogin = ({
    email,
    password,
  }: {
    email: string;
    password: string;
  }) => {
    login({ email, password });
  };

  const handleRegister = ({
    email,
    password,
  }: {
    email: string;
    password: string;
  }) => {
    register({
      email,
      username: email,
      password1: password,
      password2: password,
    });
  };

  return (
    <div className="login-container">
      <Tabs
        type="card"
        size="large"
        items={[
          {
            label: "Login",
            key: "1",
            children: (
              <Form
                form={loginForm}
                name="login"
                onFinish={handleLogin}
                initialValues={{ remember: true }}
                labelCol={{ span: 8 }}
                wrapperCol={{ span: 16 }}
              >
                <Form.Item
                  className="login-form-item"
                  label="Email"
                  name="email"
                  rules={[
                    { required: true, message: "Please input your email!" },
                  ]}
                >
                  <Input />
                </Form.Item>

                <Form.Item
                  className="login-form-item"
                  label="Password"
                  name="password"
                  rules={[
                    { required: true, message: "Please input your password!" },
                  ]}
                >
                  <Input.Password />
                </Form.Item>

                <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
                  <Button
                    className="login-button"
                    type="primary"
                    htmlType="submit"
                  >
                    Login
                  </Button>
                </Form.Item>
              </Form>
            ),
          },
          {
            label: "Register",
            key: "2",
            children: (
              <Form
                form={registerForm}
                name="register"
                onFinish={handleRegister}
                labelCol={{ span: 8 }}
                wrapperCol={{ span: 16 }}
              >
                <Form.Item
                  className="login-form-item"
                  label="Email"
                  name="email"
                  rules={[
                    { required: true, message: "Please input your email!" },
                  ]}
                >
                  <Input />
                </Form.Item>

                <Form.Item
                  className="login-form-item"
                  label="Password"
                  name="password"
                  rules={[
                    { required: true, message: "Please input your password!" },
                  ]}
                >
                  <Input.Password />
                </Form.Item>

                <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
                  <Button
                    className="login-button"
                    type="primary"
                    htmlType="submit"
                  >
                    Register
                  </Button>
                </Form.Item>
              </Form>
            ),
          },
        ]}
      />
    </div>
  );
};

export default Authentication;
