import {
  createContext,
  useCallback,
  useContext,
  useState,
  useEffect,
} from "react";
import {
  IAnalysis,
  IAnalysisRequest,
} from "../containers/Analysis/types/analysis";
import { IModels, IModelsRequest } from "../containers/Models/types/models";
import {
  ILogin,
  IRegister,
  IAuthResponse,
} from "../containers/Authentication/types/authentication";

import { mockAnalysis } from "../containers/Analysis/mocks/analysis";
import { mockModels } from "../containers/Models/mocks/models";

import { analyzeRequest } from "../services/requests/analyze";
import { modelsRequest } from "../services/requests/models";
import { loginRequest } from "../services/requests/login";
import { registerRequest } from "../services/requests/register";

import { UserStorage, AuthUser } from "../services/UserStorage";

interface AppContextState {
  analysis: IAnalysis;
  models: IModels;
  analyze: (payload: IAnalysisRequest) => void;
  runModels: (payload: IModelsRequest) => void;
  login: (payload: ILogin) => void;
  logout: () => void;
  register: (payload: IRegister) => void;
  authUser: IAuthResponse | null;
}

const defaultAppContext: AppContextState = {
  analysis: mockAnalysis,
  models: mockModels,
  analyze: (_payload: IAnalysisRequest) => {},
  runModels: (_payload: IModelsRequest) => {},
  login: (_payload: ILogin) => {},
  logout: () => {},
  register: (_payload: IRegister) => {},
  authUser: null,
};

const useAppContext = (props: AppContextState): AppContextState => {
  const [authUser, setAuthUser] = useState(props.authUser);
  const [analysis, setAnalysis] = useState(props.analysis);
  const [models, setModels] = useState(props.models);

  const updateAuthUser = useCallback((authUser: IAuthResponse | null): void => {
    setAuthUser(authUser);
  }, []);

  const updateAnalysis = useCallback((analysis: IAnalysis): void => {
    setAnalysis(analysis);
  }, []);

  const updateModels = useCallback((models: IModels): void => {
    setModels(models);
  }, []);

  const analyze = (payload: IAnalysisRequest) => {
    analyzeRequest(payload).then((data: IAnalysis) => {
      updateAnalysis(data);
    });
  };

  const runModels = (payload: IModelsRequest) => {
    modelsRequest(payload).then((data: IModels) => {
      updateModels(data);
    });
  };

  const login = (payload: ILogin) => {
    loginRequest(payload).then((data: IAuthResponse) => {
      UserStorage.storeUser(data);
      updateAuthUser(data);
    });
  };

  const register = (payload: IRegister) => {
    registerRequest(payload).then((data: IAuthResponse) => {
      UserStorage.storeUser(data);
      updateAuthUser(data);
    });
  };

  const logout = () => {
    UserStorage.clear();
    updateAuthUser(null);
    updateAnalysis({} as IAnalysis);
    updateModels({} as IModels);
  };

  useEffect(() => {
    if (UserStorage.isAuthenticated()) {
      const id = UserStorage.getUser(AuthUser.ID);
      const username = UserStorage.getUser(AuthUser.Username);
      const email = UserStorage.getUser(AuthUser.Email);
      const accessToken = UserStorage.getUser(AuthUser.AccessToken);
      const refreshToken = UserStorage.getUser(AuthUser.RefreshToken);
      if (id && username && email && accessToken) {
        updateAuthUser({
          access: accessToken,
          refresh: refreshToken || "",
          user: { pk: parseInt(id, 10), username, email },
        });
      }
    }
  }, []);

  return {
    analysis,
    models,
    analyze,
    runModels,
    login,
    logout,
    register,
    authUser,
  };
};

const AppContext = createContext<AppContextState>(defaultAppContext);

export const useApp = (): AppContextState => {
  return useContext(AppContext);
};

export const AppContextConsumer = AppContext.Consumer;

export const AppContextProvider = ({
  children,
}: {
  children: React.ReactElement;
}) => {
  const {
    analysis,
    models,
    analyze,
    runModels,
    login,
    logout,
    register,
    authUser,
  } = useAppContext(defaultAppContext);

  return (
    <AppContext.Provider
      value={{
        analysis,
        models,
        analyze,
        runModels,
        login,
        logout,
        register,
        authUser,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};
