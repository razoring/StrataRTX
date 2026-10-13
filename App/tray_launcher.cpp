#define UNICODE
#define _UNICODE
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#include <tlhelp32.h>
#include <winhttp.h>
#include <string>
#include <vector>

#pragma comment(lib, "winhttp.lib")

#define WM_TRAYICON (WM_USER + 1)
#define ID_TRAY_OPEN_WEBUI 1001
#define ID_TRAY_RESTART    1002
#define ID_TRAY_VIEW_LOG   1003
#define ID_TRAY_CONFIG     1004
#define ID_TRAY_EXIT       1005
#define IDI_STRATA_ICON    101

HINSTANCE g_hInstance = NULL;
HWND g_hWnd = NULL;
NOTIFYICONDATAW g_nid = { 0 };
PROCESS_INFORMATION g_serverPi = { 0 };

void KillProcessTree(DWORD pid) {
    if (pid == 0) return;
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnapshot != INVALID_HANDLE_VALUE) {
        PROCESSENTRY32W pe;
        pe.dwSize = sizeof(pe);
        if (Process32FirstW(hSnapshot, &pe)) {
            do {
                if (pe.th32ParentProcessID == pid) {
                    KillProcessTree(pe.th32ProcessID);
                }
            } while (Process32NextW(hSnapshot, &pe));
        }
        CloseHandle(hSnapshot);
    }
    HANDLE hProc = OpenProcess(PROCESS_TERMINATE, FALSE, pid);
    if (hProc) {
        TerminateProcess(hProc, 0);
        CloseHandle(hProc);
    }
}

void RequestHttpShutdown() {
    HINTERNET hSession = WinHttpOpen(L"StrataTray/1.0", WINHTTP_ACCESS_TYPE_NO_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (hSession) {
        HINTERNET hConnect = WinHttpConnect(hSession, L"127.0.0.1", 8080, 0);
        if (hConnect) {
            HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"POST", L"/shutdown", NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
            if (hRequest) {
                LPCWSTR headers = L"Content-Type: application/json\r\nOrigin: http://127.0.0.1:8080";
                const char* body = "{}";
                WinHttpSendRequest(hRequest, headers, (DWORD)-1L, (LPVOID)body, (DWORD)strlen(body), (DWORD)strlen(body), 0);
                WinHttpReceiveResponse(hRequest, NULL);
                WinHttpCloseHandle(hRequest);
            }
            WinHttpCloseHandle(hConnect);
        }
        WinHttpCloseHandle(hSession);
    }
}

bool RequestHttpRestart() {
    bool ok = false;
    HINTERNET hSession = WinHttpOpen(L"StrataTray/1.0", WINHTTP_ACCESS_TYPE_NO_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (hSession) {
        HINTERNET hConnect = WinHttpConnect(hSession, L"127.0.0.1", 8080, 0);
        if (hConnect) {
            HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"POST", L"/restart", NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
            if (hRequest) {
                // Increase timeouts for engine reload (up to 120s)
                WinHttpSetTimeouts(hRequest, 5000, 5000, 120000, 120000);
                LPCWSTR headers = L"Content-Type: application/json\r\nOrigin: http://127.0.0.1:8080";
                const char* body = "{}";
                if (WinHttpSendRequest(hRequest, headers, (DWORD)-1L, (LPVOID)body, (DWORD)strlen(body), (DWORD)strlen(body), 0) &&
                    WinHttpReceiveResponse(hRequest, NULL)) {
                    DWORD statusCode = 0;
                    DWORD size = sizeof(statusCode);
                    if (WinHttpQueryHeaders(hRequest, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                                            WINHTTP_HEADER_NAME_BY_INDEX, &statusCode, &size, WINHTTP_NO_HEADER_INDEX)) {
                        if (statusCode == 200) {
                            ok = true;
                        }
                    }
                }
                WinHttpCloseHandle(hRequest);
            }
            WinHttpCloseHandle(hConnect);
        }
        WinHttpCloseHandle(hSession);
    }
    return ok;
}

void StopServer() {
    RequestHttpShutdown();
    Sleep(500);

    if (g_serverPi.hProcess != NULL) {
        KillProcessTree(g_serverPi.dwProcessId);
        WaitForSingleObject(g_serverPi.hProcess, 1500);
        CloseHandle(g_serverPi.hProcess);
        CloseHandle(g_serverPi.hThread);
        ZeroMemory(&g_serverPi, sizeof(g_serverPi));
    }

    // Force-kill any lingering strata.exe instances
    STARTUPINFOW si = { sizeof(si) };
    PROCESS_INFORMATION pi = { 0 };
    WCHAR cmd[] = L"taskkill /F /IM strata.exe /T";
    if (CreateProcessW(NULL, cmd, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
        WaitForSingleObject(pi.hProcess, 2000);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }
}

bool StartServer() {
    StopServer();

    WCHAR appDir[MAX_PATH];
    GetModuleFileNameW(NULL, appDir, MAX_PATH);
    WCHAR* lastSlash = wcsrchr(appDir, L'\\');
    if (lastSlash) *lastSlash = L'\0';

    std::wstring pythonPath = std::wstring(appDir) + L"\\.venv\\Scripts\\python.exe";
    std::wstring serverPy = std::wstring(appDir) + L"\\serve\\server.py";
    std::wstring configFile = std::wstring(appDir) + L"\\strata-coder-iq1_m.json";

    std::wstring cmdLine = L"\"" + pythonPath + L"\" \"" + serverPy + L"\" --engine strata --config \"" + configFile + L"\" --port 8080";

    SetEnvironmentVariableW(L"STRATA_ARENA_PIN_GIB", L"8");
    SetEnvironmentVariableW(L"CUDA_DEVICE_ORDER", L"PCI_BUS_ID");
    SetEnvironmentVariableW(L"STRATA_STAGE_TRIM", L"1");
    SetEnvironmentVariableW(L"CUDA_P2P_DISABLE", L"1");

    STARTUPINFOW si = { sizeof(si) };
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;

    std::vector<WCHAR> cmdVec(cmdLine.begin(), cmdLine.end());
    cmdVec.push_back(L'\0');

    BOOL ok = CreateProcessW(
        NULL,
        cmdVec.data(),
        NULL,
        NULL,
        FALSE,
        CREATE_NO_WINDOW,
        NULL,
        appDir,
        &si,
        &g_serverPi
    );

    return ok != FALSE;
}

DWORD WINAPI RestartThreadProc(LPVOID lpParam) {
    // 1. Try warm live restart via HTTP endpoint (reloads weights without killing web server)
    if (RequestHttpRestart()) {
        return 0;
    }
    // 2. Fall back to cold restart
    StartServer();
    return 0;
}

void RestartServer() {
    CreateThread(NULL, 0, RestartThreadProc, NULL, 0, NULL);
}

void OpenWebUI() {
    ShellExecuteW(NULL, L"open", L"http://127.0.0.1:8080/", NULL, NULL, SW_SHOWNORMAL);
}

void OpenLog() {
    WCHAR appDir[MAX_PATH];
    GetModuleFileNameW(NULL, appDir, MAX_PATH);
    WCHAR* lastSlash = wcsrchr(appDir, L'\\');
    if (lastSlash) *lastSlash = L'\0';
    std::wstring logPath = std::wstring(appDir) + L"\\strata-coder-iq1_m.log";
    ShellExecuteW(NULL, L"open", logPath.c_str(), NULL, NULL, SW_SHOWNORMAL);
}

void OpenConfig() {
    WCHAR appDir[MAX_PATH];
    GetModuleFileNameW(NULL, appDir, MAX_PATH);
    WCHAR* lastSlash = wcsrchr(appDir, L'\\');
    if (lastSlash) *lastSlash = L'\0';
    std::wstring cfgPath = std::wstring(appDir) + L"\\strata-coder-iq1_m.json";
    ShellExecuteW(NULL, L"open", cfgPath.c_str(), NULL, NULL, SW_SHOWNORMAL);
}

void ShowTrayMenu(HWND hWnd) {
    POINT pt;
    GetCursorPos(&pt);
    HMENU hMenu = CreatePopupMenu();
    InsertMenuW(hMenu, -1, MF_BYPOSITION | MF_STRING, ID_TRAY_OPEN_WEBUI, L"Open WebUI (http://127.0.0.1:8080)");
    InsertMenuW(hMenu, -1, MF_BYPOSITION | MF_STRING, ID_TRAY_RESTART, L"Restart Strata Engine");
    InsertMenuW(hMenu, -1, MF_BYPOSITION | MF_STRING, ID_TRAY_VIEW_LOG, L"View Engine Log");
    InsertMenuW(hMenu, -1, MF_BYPOSITION | MF_STRING, ID_TRAY_CONFIG, L"Edit Config JSON");
    InsertMenuW(hMenu, -1, MF_BYPOSITION | MF_SEPARATOR, 0, NULL);
    InsertMenuW(hMenu, -1, MF_BYPOSITION | MF_STRING, ID_TRAY_EXIT, L"Exit Strata");

    SetMenuDefaultItem(hMenu, ID_TRAY_OPEN_WEBUI, FALSE);
    SetForegroundWindow(hWnd);
    TrackPopupMenu(hMenu, TPM_RIGHTBUTTON | TPM_BOTTOMALIGN, pt.x, pt.y, 0, hWnd, NULL);
    DestroyMenu(hMenu);
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_TRAYICON:
        if (lParam == WM_RBUTTONUP || lParam == WM_CONTEXTMENU) {
            ShowTrayMenu(hWnd);
        } else if (lParam == WM_LBUTTONDBLCLK) {
            OpenWebUI();
        }
        return 0;
    case WM_COMMAND:
        switch (LOWORD(wParam)) {
        case ID_TRAY_OPEN_WEBUI:
            OpenWebUI();
            break;
        case ID_TRAY_RESTART:
            RestartServer();
            break;
        case ID_TRAY_VIEW_LOG:
            OpenLog();
            break;
        case ID_TRAY_CONFIG:
            OpenConfig();
            break;
        case ID_TRAY_EXIT:
            DestroyWindow(hWnd);
            break;
        }
        return 0;
    case WM_DESTROY:
        Shell_NotifyIconW(NIM_DELETE, &g_nid);
        StopServer();
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hWnd, msg, wParam, lParam);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    HANDLE hMutex = CreateMutexW(NULL, TRUE, L"Local\\StrataTrayServerInstanceMutex");
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        OpenWebUI();
        if (hMutex) CloseHandle(hMutex);
        return 0;
    }

    g_hInstance = hInstance;

    WNDCLASSEXW wc = { sizeof(wc) };
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"StrataTrayWindowClass";
    HICON hIcon = (HICON)LoadImageW(hInstance, MAKEINTRESOURCEW(IDI_STRATA_ICON), IMAGE_ICON, 0, 0, LR_DEFAULTSIZE);
    if (!hIcon) {
        WCHAR icoPath[MAX_PATH];
        GetModuleFileNameW(NULL, icoPath, MAX_PATH);
        WCHAR* slash = wcsrchr(icoPath, L'\\');
        if (slash) *slash = L'\0';
        std::wstring diskIco = std::wstring(icoPath) + L"\\strata.ico";
        hIcon = (HICON)LoadImageW(NULL, diskIco.c_str(), IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE);
    }
    if (!hIcon) hIcon = LoadIconW(NULL, IDI_APPLICATION);
    wc.hIcon = hIcon;
    RegisterClassExW(&wc);

    g_hWnd = CreateWindowExW(0, L"StrataTrayWindowClass", L"Strata AI Tray Server", 0, 0, 0, 0, 0, NULL, NULL, hInstance, NULL);

    ZeroMemory(&g_nid, sizeof(g_nid));
    g_nid.cbSize = sizeof(g_nid);
    g_nid.hWnd = g_hWnd;
    g_nid.uID = 1;
    g_nid.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
    g_nid.uCallbackMessage = WM_TRAYICON;
    g_nid.hIcon = hIcon;
    wcsncpy_s(g_nid.szTip, L"Strata AI Server (Dual RTX 3060)", _TRUNCATE);
    Shell_NotifyIconW(NIM_ADD, &g_nid);

    StartServer();

    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    if (hMutex) {
        ReleaseMutex(hMutex);
        CloseHandle(hMutex);
    }
    return (int)msg.wParam;
}
