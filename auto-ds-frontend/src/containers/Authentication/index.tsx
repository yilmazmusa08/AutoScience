import React from "react";
import { Form, Input, Button } from "antd";
import { useApp } from "../../context/app.context";
import "./index.css";

const Login: React.FC = () => {
  const [form] = Form.useForm();
  const { login } = useApp();

  const onFinish = ({
    email,
    password,
  }: {
    email: string;
    password: string;
  }) => {
    login({ email, password });
  };

  return (
    <div className="login-container">
      <h2 className="login-title">Login</h2>
      <Form
        form={form}
        name="basic"
        onFinish={onFinish}
        initialValues={{ remember: true }}
      >
        <Form.Item
          className="login-form-item"
          label="Email"
          name="email"
          rules={[{ required: true, message: "Please input your email!" }]}
        >
          <Input />
        </Form.Item>

        <Form.Item
          className="login-form-item"
          label="Password"
          name="password"
          rules={[{ required: true, message: "Please input your password!" }]}
        >
          <Input.Password />
        </Form.Item>

        <Form.Item>
          <Button className="login-button" type="primary" htmlType="submit">
            Login
          </Button>
        </Form.Item>
      </Form>
    </div>
  );
};

export default Login;
