import React, { useState } from "react";
import { ConfigProvider, Layout, Row, Col, Menu } from "antd";
import Header from "./containers/Header";
import Authentication from "./containers/Authentication";
import Preprocessing from "./containers/Preprocessing";
import Analysis from "./containers/Analysis";
import Models from "./containers/Models";
import { useApp } from "./context/app.context";
import {
  FileDoneOutlined,
  LineChartOutlined,
  PlayCircleOutlined,
} from "@ant-design/icons";

const { Content, Sider } = Layout;

const App: React.FC = () => {
  const { authUser } = useApp();
  const [collapsed, setCollapsed] = useState(false);
  const [selectedKey, setSelectedKey] = useState("1");

  return (
    <ConfigProvider
      theme={{
        components: {
          Layout: {
            headerColor: "#fff",
            ...(authUser && { headerPadding: "0 30px 0 0" }),
          },
        },
      }}
    >
      <Layout className="layout" style={{ height: "100%" }}>
        {authUser && (
          <Sider trigger={null} collapsible collapsed={collapsed}>
            <div className="demo-logo-vertical" />
            <Menu
              theme="dark"
              mode="inline"
              defaultSelectedKeys={["1"]}
              selectedKeys={[selectedKey]}
              onSelect={({ key }) => setSelectedKey(key)}
              items={[
                {
                  key: "1",
                  icon: <FileDoneOutlined />,
                  label: "Preprocessing",
                },
                {
                  key: "2",
                  icon: <LineChartOutlined />,
                  label: "Analysis",
                },
                {
                  key: "3",
                  icon: <PlayCircleOutlined />,
                  label: "Models",
                },
              ]}
            />
          </Sider>
        )}
        <Layout>
          <Header collapsed={collapsed} setCollapsed={setCollapsed} />
          <Content style={{ padding: "0 50px" }}>
            {!authUser ? (
              <Authentication />
            ) : (
              <>
                {selectedKey === "1" ? (
                  <Preprocessing />
                ) : selectedKey === "2" ? (
                  <Analysis />
                ) : (
                  selectedKey === "3" && <Models />
                )}
              </>
            )}
          </Content>
        </Layout>
      </Layout>
    </ConfigProvider>
  );
};

export default App;
