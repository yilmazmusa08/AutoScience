import React from "react";
import { Layout, Button, Row, Col } from "antd";
import { useApp } from "../../context/app.context";
import { MenuFoldOutlined, MenuUnfoldOutlined } from "@ant-design/icons";
import "./index.css";

const { Header } = Layout;

interface HeadProps {
  collapsed: boolean;
  setCollapsed: (collapsed: boolean) => void;
}

const Head: React.FC<HeadProps> = ({ collapsed, setCollapsed }) => {
  const { logout, authUser } = useApp();

  return (
    <Header>
      <Row justify="space-between">
        {authUser && (
          <Col>
            <Button
              type="text"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed(!collapsed)}
              style={{
                fontSize: "16px",
                width: 64,
                height: 64,
                color: "white",
              }}
            />
          </Col>
        )}
        <Col className="header-title">AutoVision</Col>
        {authUser && (
          <Col>
            <Row justify="space-between" gutter={8}>
              <Col className="user-info">{`Welcome ${authUser?.user?.username}!`}</Col>
              <Col className="logout-button">
                <Button type="primary" danger onClick={logout}>
                  Logout
                </Button>
              </Col>
            </Row>
          </Col>
        )}
      </Row>
    </Header>
  );
};

export default Head;
